import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel
from tavily import TavilyClient

from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools import tool
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
# Wrapping tools separately allows proper tool-level tracing in Langfuse
# ============================================================================

@observe(as_type="tool", name="tavily-web-search")
def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web for travel information using Tavily.
    
    Args:
        query: The search query for finding travel information.
        max_results: Maximum number of results to return (default: 3).
    
    Returns:
        A formatted string containing the search results.
    """
    tavily_client = TavilyClient()
    response = tavily_client.search(query=query, max_results=max_results)
    
    # Format results for the agent
    results = []
    for result in response.get("results", []):
        results.append(f"- {result.get('title', 'No title')}: {result.get('content', 'No content')}")
    
    return "\n".join(results) if results else "No results found."

# Register as Agno tool
@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """Search the web for travel information."""
    return search_web(query, max_results)

# ============================================================================
# STRUCTURED OUTPUT SCHEMAS FOR SPECIALIZED AGENTS
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


class DailyItinerary(BaseModel):
    destination: str
    trip_duration: str
    day_by_day_plan: str

# ============================================================================
# CREATING SPECIALIZED AGENTS
# Each agent is wrapped with @observe(as_type="agent") when executed
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

itinerary_planner = Agent(
    id="itinerary-planner",
    name="Itinerary Planner",
    role="Expert at creating detailed day-by-day travel itineraries",
    description="You are a travel itinerary specialist who creates well-organized, realistic day-by-day plans.",
    instructions=[
        "Create a logical day-by-day schedule",
        "Group nearby attractions together to minimize travel time",
        "Include time for meals, rest, and flexibility",
        "Balance activities with relaxation",
    ],
    model=OpenAIChat(id="gpt-4.1-mini"),
    output_schema=DailyItinerary,
    markdown=True,
)

# ============================================================================
# WRAP AGENT RUN METHODS with @observe for proper tracing
# This enables individual agent tracing when the Team orchestrates them
# ============================================================================

def make_agent_observable(agent: Agent, agent_name: str) -> None:
    """
    Wraps an agent's run and arun methods with Langfuse @observe decorator.
    
    This allows proper tracing when the Agno Team internally delegates
    work to individual agents - each agent execution becomes a separate
    observable span in Langfuse.
    
    How it works:
    1. Store a reference to the original agent.run/arun methods (no execution yet)
    2. Create wrapper functions decorated with @observe
    3. Replace agent.run/arun with the wrappers
    4. When Team calls agent.run()/arun() → wrapper executes → original method called once
    
    Args:
        agent: The Agno Agent instance to wrap
        agent_name: The name to use for the observation in Langfuse traces
    """
    # Wrap synchronous run method
    original_run_method = agent.run
    
    @observe(as_type="agent", name=agent_name)
    def run_with_observation(*args, **kwargs):
        """Wrapper that traces the agent execution."""
        return original_run_method(*args, **kwargs)
    
    agent.run = run_with_observation  # type: ignore[method-assign]
    
    # Wrap asynchronous arun method
    original_arun_method = agent.arun
    
    @observe(as_type="agent", name=agent_name)
    async def arun_with_observation(*args, **kwargs):
        """Async wrapper that traces the agent execution."""
        return await original_arun_method(*args, **kwargs)
    
    agent.arun = arun_with_observation  # type: ignore[method-assign]


# Apply Langfuse observation to each specialized agent
# This ensures each agent's execution is properly traced when called by the Team
make_agent_observable(destination_researcher, "destination-researcher")
make_agent_observable(hotel_finder, "hotel-finder")
make_agent_observable(itinerary_planner, "itinerary-planner")

# ============================================================================
# TRAVEL PLANNING TEAM (Agno Team approach with async execution)
# Team delegates to all members simultaneously for parallel execution
# ============================================================================

travel_team = Team(
    name="Travel Planning Team",
    members=[destination_researcher, hotel_finder, itinerary_planner],
    model=OpenAIChat(id="gpt-4.1-mini"),
    instructions=[
        "You are a travel planning manager coordinating a team of specialists",
        "Delegate tasks to multiple team members simultaneously when possible",
        "Ask the Destination Researcher to gather info about the destination",
        "Ask the Hotel Finder to find accommodation options", 
        "Ask the Itinerary Planner to create a day-by-day schedule based on the findings",

        "You will also act as the final report writer, combining all findings into a comprehensive, easy-to-read travel plan",
        "The final report should be in markdown format with the following format:",
        "# Comprehensive Travel Plan: [Destination] [Amount of Days] Day Trip",
        "## Destination Overview",
        "## Accommodation Recommendations",
        "### [Amount of Days] Day Itinerary",
        "### Additional Notes and Travel Tips",
        "### Summary Table of the Day-by-Day Plan",
    ],
    show_members_responses=True,
    markdown=True,
    
    # ⭐ KEY: Delegate to all members at once for parallel execution
    delegate_to_all_members=True,

    # # Enable debug mode for detailed events execution in terminal
    # debug_mode=True
)

@observe(as_type="agent", name="travel-planning-team")
async def run_travel_team(query: str):
    """Run the travel planning team asynchronously for concurrent agent execution."""
    response = await travel_team.arun(query)  # ⭐ Use arun() for concurrent execution
    return response

# ============================================================================
# MAIN PIPELINE with @observe
# ============================================================================

@observe(as_type="span", name="Travel Planning Pipeline")
async def plan_trip(query: str):
    """
    Run the travel planning team with proper Langfuse tracing.
    
    With async + delegate_to_all_members=True:
    - destination-researcher and hotel-finder run CONCURRENTLY (parallel)
    - itinerary-planner may run after or concurrently (depends on Team's decision)
    
    Travel Planning Pipeline (span)
    └── travel-planning-team (agent)
        ├── destination-researcher (agent) ⚡ concurrent
        │   ├── tavily-web-search (tool)
        │   └── openai.chat.completions (generation)
        ├── hotel-finder (agent) ⚡ concurrent
        │   ├── tavily-web-search (tool)
        │   └── openai.chat.completions (generation)
        └── itinerary-planner (agent)
            └── openai.chat.completions (generation)
    """
    with propagate_attributes(
        user_id="cikalmerdeka",
        session_id="1234",
        tags=["travel", "planning", "async"],
        version="1.0.0",
        metadata={
            "experiment": "variant_a",
            "environment": "development",
            "execution_mode": "async_parallel"
        }
    ):

        result = await run_travel_team(query)

        # Update trace with final input/output
        langfuse.update_current_trace(
            input=query,
            output=result.content if result else None,
        )

        return result

# Use the team
if __name__ == "__main__":
    
    # Define the query
    query = "Plan a 5-day trip to Kyoto, Japan for a solo traveler interested in temples, traditional culture, and local food. Budget is mid-range."

    # Run the travel planning team asynchronously for concurrent execution
    result = asyncio.run(plan_trip(query))
    
    # Print the result
    pprint_run_response(result, markdown=True)

    langfuse.flush()  # Ensure traces are sent before script exits

