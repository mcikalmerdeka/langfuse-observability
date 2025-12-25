import asyncio
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tavily import TavilyClient

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.run import RunContext
from agno.workflow import Workflow, Step, Loop, Parallel, Router
from agno.workflow.types import StepInput, StepOutput
from agno.utils.pprint import pprint_run_response
from langfuse import get_client, observe, propagate_attributes
import openlit


# Load environment variables
load_dotenv()

# Initialize Langfuse client
langfuse = get_client()

# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

# Initialize OpenLIT instrumentation
openlit.init(tracer=langfuse._otel_tracer, disable_batch=True)

# ============================================================================
# CUSTOM TOOLS with @observe(as_type="tool")
# ============================================================================

@observe(as_type="tool", name="tavily-web-search")
def search_web(query: str, max_results: int = 3) -> str:
    """Search the web for travel information using Tavily."""
    tavily_client = TavilyClient()
    response = tavily_client.search(query=query, max_results=max_results)
    
    results = []
    for result in response.get("results", []):
        results.append(f"- {result.get('title', 'No title')}: {result.get('content', 'No content')}")
    
    return "\n".join(results) if results else "No results found."

@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """Search the web for travel information."""
    return search_web(query, max_results)

# ============================================================================
# STRUCTURED OUTPUT SCHEMAS
# ============================================================================

class DestinationInfo(BaseModel):
    destination: str
    top_attractions: str
    best_time_to_visit: str
    local_tips: str
    weather_info: str


class AccommodationOptions(BaseModel):
    destination: str
    hotel_recommendations: str
    budget_range: str
    booking_tips: str


class ActivitiesInfo(BaseModel):
    destination: str
    recommended_activities: str
    transportation_options: str
    local_experiences: str
    estimated_costs: str


class DailyItinerary(BaseModel):
    destination: str
    trip_duration: str
    day_by_day_plan: str


class CritiqueResult(BaseModel):
    is_approved: bool
    overall_assessment: str
    agents_to_rerun: List[str]  # List of agent names to rerun: "destination", "hotel", "activities"
    specific_feedback: str
    improvement_suggestions: str

# ============================================================================
# SPECIALIZED RESEARCH AGENTS (3 agents for parallel execution)
# ============================================================================

destination_researcher = Agent(
    id="destination-researcher",
    name="Destination Researcher",
    role="Expert at researching travel destinations",
    description="You are a travel expert who finds the best attractions, local tips, weather info, and cultural insights for destinations.",
    instructions=[
        "Search for top attractions and must-see places",
        "Find current weather conditions and best times to visit",
        "Discover local tips, cultural etiquette, and hidden gems",
        "Focus on practical, up-to-date information",
    ],
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[web_search_tool],
    output_schema=DestinationInfo,
    markdown=True,
)

hotel_finder = Agent(
    id="hotel-finder",
    name="Hotel & Accommodation Finder",
    role="Expert at finding the best hotels and accommodations",
    description="You are a travel accommodation specialist who finds the best places to stay based on budget and preferences.",
    instructions=[
        "Search for highly-rated hotels and accommodations",
        "Consider different budget ranges (budget, mid-range, luxury)",
        "Look for good locations near attractions",
        "Provide booking tips and best times to book",
    ],
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[web_search_tool],
    output_schema=AccommodationOptions,
    markdown=True,
)

activities_researcher = Agent(
    id="activities-researcher",
    name="Activities & Experiences Researcher",
    role="Expert at finding local activities, transportation, and unique experiences",
    description="You are a travel activities specialist who finds the best things to do, local transportation options, and unique experiences.",
    instructions=[
        "Search for popular activities and unique local experiences",
        "Find transportation options (public transit, car rental, walking routes)",
        "Discover food tours, cultural workshops, and authentic local experiences",
        "Provide estimated costs for activities and transportation",
    ],
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[web_search_tool],
    output_schema=ActivitiesInfo,
    markdown=True,
)

# ============================================================================
# ITINERARY PLANNER AGENT
# ============================================================================

itinerary_planner = Agent(
    id="itinerary-planner",
    name="Middle Manager - Itinerary Planner",
    role="Middle manager who synthesizes team research into comprehensive travel plans",
    description="You are a middle manager who takes research from your team and creates executive-ready travel plans for VP approval.",
    instructions=[
        "You receive research from your team (destination, hotel, and activities researchers)",
        "Your job is to synthesize this information into a comprehensive, polished travel plan",
        "",
        "When creating the initial plan:",
        "- Review ALL research from the destination, hotel, and activities team members",
        "- Synthesize their findings into a cohesive narrative",
        "- Create a logical day-by-day schedule balancing activities with rest",
        "- Group nearby attractions to minimize travel time",
        "",
        "When revising based on VP feedback:",
        "- Carefully read the VP's critique and address ALL points raised",
        "- Improve structure, clarity, and completeness as requested",
        "- Work with the EXISTING research data - don't make up new information",
        "- Polish the presentation for executive approval",
        "",
        "When presenting the final approved plan:",
        "- Present the VP-approved version as the final deliverable",
        "- Make it user-friendly and ready for the traveler to use",
        "- Ensure it's comprehensive and professional",
        "",
        "Output format (comprehensive markdown report):",
        "# Comprehensive Travel Plan: [Destination] [Amount of Days] Day Trip",
        "## Executive Summary",
        "## Destination Overview (from destination research)",
        "## Accommodation Recommendations (from hotel research)",
        "## Activities & Experiences (from activities research)",
        "## Day-by-Day Itinerary",
        "## Transportation Guide",
        "## Budget Breakdown",
        "## Additional Notes and Travel Tips",
    ],
    model=OpenAIChat(id="gpt-4.1-mini"),
    markdown=True,
)

# ============================================================================
# CRITIQUE/REVISION AGENT (using gpt-4.1 for more thorough analysis)
# ============================================================================

critique_agent = Agent(
    id="critique-agent",
    name="VP - Travel Plan Reviewer",
    role="Senior executive who reviews and approves travel plans",
    description="You are a VP-level reviewer who evaluates travel plans for quality and completeness before final approval.",
    instructions=[
        "You are reviewing a travel plan prepared by your middle manager (itinerary planner)",
        "The middle manager has already synthesized research from the team (destination, hotel, activities researchers)",
        "Your role is to provide executive-level feedback:",
        "",
        "Evaluate the plan for:",
        "1. Completeness - Does it cover all essential aspects?",
        "2. Coherence - Does the itinerary flow logically?",
        "3. Practicality - Are activities realistic within the timeframe?",
        "4. Value - Does it align with the budget and traveler preferences?",
        "",
        "Decision criteria:",
        "- On FIRST review: Be thorough but constructive. Approve if solid, or request specific improvements",
        "- On SECOND review (if revision occurred): Be more lenient and approve if reasonably good",
        "",
        "When requesting revisions:",
        "- Focus on how the REPORT should be improved (structure, clarity, completeness)",
        "- Don't ask for new research - work with existing team data",
        "- Give specific, actionable feedback the middle manager can implement",
    ],
    model=OpenAIChat(id="gpt-4.1"),  # Using more capable model for executive-level critique
    output_schema=CritiqueResult,
    markdown=True,
)

# ============================================================================
# WRAP AGENT RUN METHODS with @observe for proper Langfuse tracing
# ============================================================================

def make_agent_observable(agent: Agent, agent_name: str) -> None:
    """Wraps an agent's run and arun methods with Langfuse @observe decorator."""
    original_run_method = agent.run
    
    @observe(as_type="agent", name=agent_name)
    def run_with_observation(*args, **kwargs):
        return original_run_method(*args, **kwargs)
    
    agent.run = run_with_observation  # type: ignore[method-assign]
    
    original_arun_method = agent.arun
    
    @observe(as_type="agent", name=agent_name)
    async def arun_with_observation(*args, **kwargs):
        return await original_arun_method(*args, **kwargs)
    
    agent.arun = arun_with_observation  # type: ignore[method-assign]


# Apply Langfuse observation to all agents
make_agent_observable(destination_researcher, "destination-researcher")
make_agent_observable(hotel_finder, "hotel-finder")
make_agent_observable(activities_researcher, "activities-researcher")
make_agent_observable(itinerary_planner, "itinerary-planner")
make_agent_observable(critique_agent, "critique-agent")

# ============================================================================
# WORKFLOW STEPS
# ============================================================================

# Initial parallel research steps
destination_step = Step(
    name="Research Destination",
    agent=destination_researcher,
    description="Research the destination's attractions, weather, and local tips"
)

hotel_step = Step(
    name="Find Accommodations",
    agent=hotel_finder,
    description="Find suitable hotels and accommodations based on budget"
)

activities_step = Step(
    name="Research Activities",
    agent=activities_researcher,
    description="Research activities, transportation, and local experiences"
)

# Itinerary planning step
itinerary_step = Step(
    name="Create Itinerary",
    agent=itinerary_planner,
    description="Create a comprehensive day-by-day travel itinerary"
)

# ============================================================================
# CUSTOM FUNCTION STEPS for Critique Logic
# ============================================================================

def critique_and_revise(step_input: StepInput, run_context: RunContext) -> StepOutput:  # type: ignore[arg-type]
    """
    Critique agent reviews the itinerary and provides feedback.
    This function stores the critique feedback for the itinerary planner to use.
    """
    # Ensure session_state is initialized
    if run_context.session_state is None:
        run_context.session_state = {}
    
    # Get iteration count from session state
    iteration = run_context.session_state.get("revision_iteration", 0) + 1
    run_context.session_state["revision_iteration"] = iteration
    
    print(f"\n🔍 VP Critique - Review #{iteration}/2")
    
    # Build context for critique
    itinerary_content = step_input.previous_step_content or ""
    
    critique_prompt = f"""
    You are the VP reviewing a travel plan prepared by your team.
    
    TRAVEL PLAN TO REVIEW (Draft #{iteration}):
    {itinerary_content}
    
    {"This is the FINAL review - approve if it's reasonably good." if iteration >= 2 else "This is the initial review - provide constructive feedback."}
    
    Evaluate the plan for:
    1. Completeness - Does it cover all aspects (destination info, hotels, activities, itinerary)?
    2. Coherence - Does the day-by-day plan flow logically?
    3. Practicality - Are the activities feasible within the timeframe?
    4. Budget alignment - Do recommendations match the stated budget?
    
    Provide your structured assessment with specific improvement suggestions if needed.
    """
    
    response = critique_agent.run(critique_prompt)
    
    # Parse the critique result
    if response.content:
        try:
            # Try to get structured output
            critique_data = getattr(response, 'response_model', None)
            
            if critique_data and isinstance(critique_data, CritiqueResult):
                is_approved = critique_data.is_approved
            else:
                # Fallback: parse from content
                content_lower = str(response.content).lower()
                is_approved = ("approved" in content_lower or "good" in content_lower) and "not approved" not in content_lower
            
            # Store in session state
            run_context.session_state["critique_approved"] = is_approved
            run_context.session_state["last_critique"] = response.content
            
            status = "✅ APPROVED" if is_approved else "🔄 NEEDS REVISION"
            print(f"   VP Decision: {status}")
            
            return StepOutput(
                content=response.content,
                success=True
            )
        except Exception as e:
            print(f"   ⚠️ Error parsing critique: {e}")
            # Auto-approve after 2 iterations
            run_context.session_state["critique_approved"] = iteration >= 2
            return StepOutput(content=str(response.content), success=True)
    
    # Default: approve if we've done 2 iterations
    run_context.session_state["critique_approved"] = iteration >= 2
    return StepOutput(content="Critique completed", success=True)


# Custom step for critique
critique_step = Step(
    name="VP Critique",
    executor=critique_and_revise,  # type: ignore[arg-type]
    description="VP reviews the travel plan and provides approval or revision feedback"
)

# ============================================================================
# LOOP END CONDITION
# ============================================================================

def revision_approved_condition(outputs: List[StepOutput], run_context: RunContext) -> bool:  # type: ignore[arg-type]
    """
    End condition for the revision loop between itinerary planner and VP critique.
    Returns True to BREAK the loop (when approved or max iterations reached), False to continue.
    """
    # Ensure session_state is initialized
    if run_context.session_state is None:
        run_context.session_state = {}
    
    is_approved: bool = run_context.session_state.get("critique_approved", False)
    iteration: int = run_context.session_state.get("revision_iteration", 0)
    
    if is_approved:
        print(f"\n✅ Travel plan APPROVED by VP after {iteration} iteration(s)!")
        return True
    
    if iteration >= 2:
        print(f"\n⚠️ Max iterations reached ({iteration}). Finalizing plan.")
        return True
    
    print(f"\n🔄 Middle Manager revising based on VP feedback...")
    return False

# ============================================================================
# FINAL REPORT STEP
# ============================================================================

# Create a final step where itinerary planner presents the approved report
final_report_step = Step(
    name="Present Final Report",
    agent=itinerary_planner,
    description="Middle Manager presents the VP-approved travel plan to the user"
)

# ============================================================================
# COMBINED WORKFLOW - Simplified Architecture
# ============================================================================

travel_planning_workflow = Workflow(
    name="Travel Planning Workflow with VP Approval",
    description="""
    A streamlined travel planning workflow:
    1. Research Team (destination, hotel, activities) runs in parallel ONCE
    2. Middle Manager (itinerary planner) creates comprehensive report
    3. VP (critique agent) reviews and provides feedback
    4. Loop (max 1 revision): Middle Manager revises report based on VP feedback
    5. Middle Manager presents final VP-approved report to user
    """,
    # Initialize session state for tracking workflow progress
    session_state={
        "revision_iteration": 0,
        "critique_approved": False,
        "last_critique": "",
    },
    steps=[  # type: ignore[arg-type]
        # Step 1: Research Team - Parallel research phase (runs ONCE only)
        Parallel(
            destination_step,  # type: ignore[list-item]
            hotel_step,  # type: ignore[list-item]
            activities_step,  # type: ignore[list-item]
            name="Research Team Phase",
            description="Research team gathers destination, hotel, and activities data simultaneously"
        ),
        # Step 2: Middle Manager + VP Loop (max 2 iterations: initial + 1 revision)
        Loop(
            name="Middle Manager <-> VP Revision Loop",
            description="Itinerary planner (middle manager) works with VP to finalize the plan",
            steps=[
                itinerary_step,  # type: ignore[list-item] - Middle Manager creates/revises report
                critique_step,  # type: ignore[list-item] - VP reviews and approves/requests revision
            ],
            end_condition=revision_approved_condition,  # type: ignore[arg-type]
            max_iterations=2,  # Initial draft + 1 revision max
        ),
        # Step 3: Middle Manager presents the final VP-approved report
        final_report_step,  # type: ignore[list-item] - This becomes the final output to the user
    ],
)

# ============================================================================
# MAIN PIPELINE with @observe
# ============================================================================

@observe(as_type="agent", name="travel-planning-workflow")
async def run_travel_workflow(query: str):
    """Run the travel planning workflow asynchronously."""
    response = await travel_planning_workflow.arun(query)
    return response


@observe(as_type="span", name="Travel Planning Pipeline (Combined)")
async def plan_trip(query: str):
    """
    Run the simplified travel planning workflow with Langfuse tracing.
    
    Workflow Structure:
    ==================
    Travel Planning Pipeline (span)
    └── travel-planning-workflow (agent)
        ├── Research Team Phase ⚡ (runs ONCE only)
        │   ├── destination-researcher (agent) → tavily-web-search
        │   ├── hotel-finder (agent) → tavily-web-search
        │   └── activities-researcher (agent) → tavily-web-search
        │
        ├── Middle Manager <-> VP Revision Loop (max 2 iterations)
        │   ├── Iteration 1:
        │   │   ├── itinerary-planner (agent) → creates initial report
        │   │   └── critique-agent (VP) → reviews and approves/requests revision
        │   └── Iteration 2 (if VP requested revision):
        │       ├── itinerary-planner (agent) → revises report based on VP feedback
        │       └── critique-agent (VP) → final approval
        │
        └── Present Final Report 📋
            └── itinerary-planner (agent) → presents VP-approved plan to user
    
    Benefits:
    - Research agents (Tavily tool calls) run ONLY ONCE at the start
    - Only itinerary planner revises in loop, no redundant research calls
    - Maximum 2 loop iterations = 1 revision opportunity
    - Mimics real org structure: Team → Middle Manager → VP approval
    - Final output is from itinerary planner (natural language), not critique (structured)
    """
    with propagate_attributes(
        user_id="cikalmerdeka",
        session_id="combined-workflow-001",
        tags=["travel", "planning", "workflow", "vp-approval"],
        version="2.0.0",
        metadata={
            "experiment": "simplified_workflow",
            "environment": "development",
            "execution_mode": "parallel_once_then_revision_loop"
        }
    ):
        result = await run_travel_workflow(query)
        
        # Update trace with final input/output
        langfuse.update_current_trace(
            input=query,
            output=result.content if result else None,
        )
        
        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Define the query
    query = "Plan a 5-day trip to Kyoto, Japan for a solo traveler interested in temples, traditional culture, and local food. Budget is mid-range."
    
    print("=" * 70)
    print("🌏 SIMPLIFIED TRAVEL PLANNING WORKFLOW")
    print("=" * 70)
    print(f"\n📝 Query: {query}\n")
    print("=" * 70)
    print("\n🔄 Workflow Flow:")
    print("   1. Research Team (3 agents) → Parallel research (ONE TIME ONLY)")
    print("   2. Middle Manager (itinerary planner) → Creates report")
    print("   3. VP (critique agent) → Reviews report")
    print("   4. [IF NEEDED] Middle Manager → Revises based on VP feedback")
    print("   5. VP → Final approval")
    print("   6. Middle Manager → Presents final approved plan to user 📋")
    print("=" * 70)
    
    # Run the travel planning workflow
    result = asyncio.run(plan_trip(query))
    
    # Print the result
    print("\n" + "=" * 70)
    print("📋 FINAL TRAVEL PLAN")
    print("=" * 70)
    pprint_run_response(result, markdown=True)
    
    # Ensure traces are sent before script exits
    langfuse.flush()

