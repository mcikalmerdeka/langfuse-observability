import asyncio
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from tavily import TavilyClient
from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.run import RunContext
from agno.workflow import Workflow, Step, Loop, Parallel
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
    instructions=dedent("""\
        Search for top attractions and must-see places
        Find current weather conditions and best times to visit
        Discover local tips, cultural etiquette, and hidden gems
        Focus on practical, up-to-date information
    """),
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
    instructions=dedent("""\
        Search for highly-rated hotels and accommodations
        Consider different budget ranges (budget, mid-range, luxury)
        Look for good locations near attractions
        Provide booking tips and best times to book
    """),
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
    instructions=dedent("""\
        Specify the destination name
        Search for popular activities and unique local experiences
        Find transportation options (public transit, car rental, walking routes)
        Discover food tours, cultural workshops, and authentic local experiences
        Provide estimated costs for activities and transportation
    """),
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
    name="Team Lead - Itinerary Planner",
    role="Team lead who synthesizes team research into comprehensive travel plans",
    description="You are a team lead who takes research from your team members and creates manager-ready travel plans for approval.",
    instructions=dedent("""\
        You receive research from your team (destination, hotel, and activities researchers)
        
        Your job is to synthesize this information into a comprehensive, polished travel plan
        
        CURRENT STATUS:
        - Revision iteration: {revision_iteration}
        - Manager's last feedback: {manager_feedback}
        - Previous draft: {previous_draft}
        
        When creating the initial plan (iteration 0):
        - Review ALL research from the destination, hotel, and activities team members
        - Synthesize their findings into a cohesive narrative
        - Create a logical day-by-day schedule balancing activities with rest
        - Group nearby attractions to minimize travel time
        
        When revising based on Manager feedback (iteration > 0):
        - Carefully read the Manager's critique from session state: {manager_feedback}
        - Review your previous draft: {previous_draft}
        - Address ALL points raised in the feedback
        - Improve structure, clarity, and completeness as requested
        - Work with the EXISTING research data - don't make up new information
        - Polish the presentation for manager approval
        
        When presenting the final approved plan:
        - Present the Manager-approved version as the final deliverable
        - Make it user-friendly and ready for the traveler to use
        - Ensure it's comprehensive and professional

        Output format (comprehensive markdown report):
        # Comprehensive Travel Plan: [Destination] [Amount of Days] Day Trip
        ## Executive Summary
        ## Destination Overview (from destination research)
        ## Accommodation Recommendations (from hotel research)
        ## Activities & Experiences (from activities research)
        
        ## Day-by-Day Itinerary
        IMPORTANT: Format the itinerary as a table with the following structure:
        - Column 1: Day (with bold day headers like **Day 1: Theme**)
        - Column 2: Activities & Timing (with time ranges and activity descriptions)
        - Use format: HH:MM – HH:MM | Activity description
        - Group activities by day with bold day headers in the Day column
        - Example format:
          | Day | Activities & Timing |,
          |-----|---------------------|,
          | **Day 1: Arrival & Theme** |,
          | 09:00 – 11:00 | Activity description |,
          | 11:30 – 13:00 | Next activity |,
        ## Transportation Guide
        
        ## Budget Breakdown
        IMPORTANT: Format the budget as a table with the following structure:
        | Category | Estimated Cost (JPY) | Notes |
        |----------|---------------------|-------|
        | Accommodation | 25,000/night | Hotel details |
        | Meals | 4,000/day | Dining style |
        | **Total (X days)** | **~XXX,XXX JPY** | Per person estimate |
        ## Additional Notes and Travel Tips
    """),
    model=OpenAIChat(id="gpt-4.1-mini"),
    # Enable session state for tracking revisions and feedback
    session_state={
        "revision_iteration": 0,
        "manager_feedback": "No feedback yet - this is the initial draft",
        "previous_draft": "No previous draft",
    },
    add_session_state_to_context=True,
    markdown=True,
)

# ============================================================================
# CRITIQUE/REVISION AGENT (recommended to use more capable model for managerial-level critique)
# ============================================================================

critique_agent = Agent(
    id="critique-agent",
    name="Manager - Travel Plan Reviewer",
    role="Manager who reviews and approves travel plans",
    description="You are a manager who evaluates travel plans for quality and completeness before final approval.",
    instructions=dedent("""\
        You are reviewing a travel plan prepared by your team lead (itinerary planner)
        The team lead has already synthesized research from the team (destination, hotel, activities researchers)
        Your role is to provide managerial-level feedback:
        
        CURRENT REVIEW STATUS:
        - Review iteration: {revision_iteration}
        - Previous feedback given: {manager_feedback}
        
        Evaluate the plan for:
        1. Completeness - Does it cover all essential aspects?
        2. Coherence - Does the itinerary flow logically?
        3. Practicality - Are activities realistic within the timeframe?
        4. Value - Does it align with the budget and traveler preferences?
        
        Decision criteria:
        - On FIRST review (iteration 0): Be thorough but constructive. Approve if solid, or request specific improvements
        - On SECOND review (iteration 1): Be more lenient and approve if reasonably good
        
        When requesting revisions:
        - Focus on how the REPORT should be improved (structure, clarity, completeness)
        - Don't ask for new research - work with existing team data 
        - Give specific, actionable feedback the team lead can implement
    """),
    model=OpenAIChat(id="gpt-4.1-mini"),  # Using more capable model for managerial-level critique
    output_schema=CritiqueResult,
    # Enable session state for tracking review iterations and feedback
    session_state={
        "revision_iteration": 0,
        "manager_feedback": "No feedback yet",
        "is_approved": False,
    },
    add_session_state_to_context=True,
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
    Manager reviews the itinerary and provides feedback.
    Updates session_state with critique results for the team lead to access.
    """
    # Ensure session_state is initialized
    if run_context.session_state is None:
        run_context.session_state = {}
    
    # Get the current draft from previous step
    current_draft = step_input.previous_step_content or ""
    
    # Store the current draft in session state for next revision
    run_context.session_state["previous_draft"] = current_draft
    
    # Get current iteration
    iteration = run_context.session_state.get("revision_iteration", 0)
    
    print(f"\n🔍 Manager Review - Review #{iteration + 1}/2")
    
    # Build critique prompt
    critique_prompt = f"""
    You are the Manager reviewing a travel plan prepared by your team lead.
    
    TRAVEL PLAN TO REVIEW (Draft #{iteration + 1}):
    {current_draft}
    
    {"This is the FINAL review - approve if it's reasonably good." if iteration >= 1 else "This is the initial review - provide constructive feedback."}
    
    Evaluate the plan for:
    1. Completeness - Does it cover all aspects (destination info, hotels, activities, itinerary)?
    2. Coherence - Does the day-by-day plan flow logically?
    3. Practicality - Are the activities feasible within the timeframe?
    4. Budget alignment - Do recommendations match the stated budget?
    
    Provide your structured assessment with specific improvement suggestions if needed.
    """
    
    # Run critique agent with session_state
    response = critique_agent.run(
        critique_prompt,
        session_state=run_context.session_state
    )
    
    # Parse the critique result
    is_approved = False
    feedback_text = ""
    
    if response.content:
        try:
            # Try to get structured output
            critique_data = getattr(response, 'response_model', None)
            
            if critique_data and isinstance(critique_data, CritiqueResult):
                is_approved = critique_data.is_approved
                feedback_text = f"{critique_data.overall_assessment}\n\nSpecific Feedback:\n{critique_data.specific_feedback}\n\nSuggestions:\n{critique_data.improvement_suggestions}"
            else:
                # Fallback: parse from content
                content_lower = str(response.content).lower()
                is_approved = ("approved" in content_lower or "good" in content_lower) and "not approved" not in content_lower
                feedback_text = str(response.content)
            
        except Exception as e:
            print(f"   ⚠️ Error parsing critique: {e}")
            # Auto-approve after 2 iterations
            is_approved = iteration >= 1
            feedback_text = str(response.content) if response.content else "Critique completed"
    else:
        # Default: approve if we've done 2 iterations
        is_approved = iteration >= 1
        feedback_text = "Critique completed"
    
    # Update session state with critique results
    run_context.session_state["is_approved"] = is_approved
    run_context.session_state["manager_feedback"] = feedback_text
    run_context.session_state["revision_iteration"] = iteration + 1
    
    status = "✅ APPROVED" if is_approved else "🔄 NEEDS REVISION"
    print(f"   Manager Decision: {status}")
    
    return StepOutput(
        content=feedback_text,
        success=True
    )


# Custom step for critique
critique_step = Step(
    name="Manager Review",
    executor=critique_and_revise,  # type: ignore[arg-type]
    description="Manager reviews the travel plan and provides approval or revision feedback"
)

# ============================================================================
# LOOP END CONDITION
# ============================================================================

def revision_approved_condition(outputs: List[StepOutput], run_context: RunContext) -> bool:  # type: ignore[arg-type]
    """
    End condition for the revision loop between team lead and manager.
    Returns True to BREAK the loop (when approved or max iterations reached), False to continue.
    
    Checks session_state for approval status set by the critique agent.
    """
    # Ensure session_state is initialized
    if run_context.session_state is None:
        run_context.session_state = {}
    
    # Get approval status and iteration from session state
    is_approved: bool = run_context.session_state.get("is_approved", False)
    iteration: int = run_context.session_state.get("revision_iteration", 0)
    
    if is_approved:
        print(f"\n✅ Travel plan APPROVED by Manager after {iteration} iteration(s)!")
        return True
    
    if iteration >= 2:
        print(f"\n⚠️ Max iterations reached ({iteration}). Finalizing plan.")
        return True
    
    print(f"\n🔄 Team Lead revising based on Manager feedback...")
    return False

# ============================================================================
# FINAL REPORT STEP
# ============================================================================

# Create a final step where itinerary planner presents the approved report
final_report_step = Step(
    name="Present Final Report",
    agent=itinerary_planner,
    description="Team Lead presents the Manager-approved travel plan to the user"
)

# ============================================================================
# COMBINED WORKFLOW - Travel Planning Workflow Architecture
# ============================================================================

travel_planning_workflow = Workflow(
    name="Travel Planning Workflow with Manager Approval",
    description="""
    A streamlined travel planning workflow:
    1. Research Team (destination, hotel, activities) runs in parallel ONCE
    2. Team Lead (itinerary planner) creates comprehensive report
    3. Manager (critique agent) reviews and provides feedback
    4. Loop (max 1 revision): Team Lead revises report based on Manager feedback
    5. Team Lead presents final Manager-approved report to user
    """,
    # Initialize session state for tracking workflow progress
    session_state={
        "revision_iteration": 0,
        "is_approved": False,
        "manager_feedback": "No feedback yet - this is the initial draft",
        "previous_draft": "No previous draft",
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
        # Step 2: Team Lead + Manager Loop (max 2 iterations: initial + 1 revision)
        Loop(
            name="Team Lead <-> Manager Revision Loop",
            description="Itinerary planner (team lead) works with Manager to finalize the plan",
            steps=[
                itinerary_step,  # type: ignore[list-item] - Team Lead creates/revises report
                critique_step,  # type: ignore[list-item] - Manager reviews and approves/requests revision
            ],
            end_condition=revision_approved_condition,  # type: ignore[arg-type]
            max_iterations=2,  # Initial draft + 1 revision max
        ),
        # Step 3: Team Lead presents the final Manager-approved report
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
        ├── Team Lead <-> Manager Revision Loop (max 2 iterations)
        │   ├── Iteration 1:
        │   │   ├── itinerary-planner (team lead) → creates initial report
        │   │   └── critique-agent (manager) → reviews and approves/requests revision
        │   └── Iteration 2 (if Manager requested revision):
        │       ├── itinerary-planner (team lead) → revises report based on Manager feedback
        │       └── critique-agent (manager) → final approval
        │
        └── Present Final Report 📋
            └── itinerary-planner (team lead) → presents Manager-approved plan to user
    
    Benefits:
    - Research agents (Tavily tool calls) run ONLY ONCE at the start
    - Only itinerary planner revises in loop, no redundant research calls
    - Maximum 2 loop iterations = 1 revision opportunity
    - Mimics real org structure: Research Team → Team Lead → Manager approval
    - Final output is from itinerary planner (natural language), not critique (structured)
    """
    with propagate_attributes(
        user_id="cikalmerdeka",
        session_id="combined-workflow-001",
        tags=["travel", "planning", "workflow", "manager-approval"],
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
    print("   2. Team Lead (itinerary planner) → Creates report")
    print("   3. Manager (critique agent) → Reviews report")
    print("   4. [IF NEEDED] Team Lead → Revises based on Manager feedback")
    print("   5. Manager → Final approval")
    print("   6. Team Lead → Presents final approved plan to user 📋")
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

