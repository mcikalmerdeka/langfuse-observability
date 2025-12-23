import os
from dotenv import load_dotenv
from pydantic import BaseModel

from langfuse import get_client, observe
from agno.agent import Agent
from agno.team import Team
from agno.models.openai import OpenAIChat
from agno.tools.tavily import TavilyTools
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

# Define structured output schemas for agents
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


# Create specialized agents
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
    tools=[TavilyTools()],
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
    tools=[TavilyTools()],
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

# Create the Travel Planning Team
travel_team = Team(
    name="Travel Planning Team",
    members=[destination_researcher, hotel_finder, itinerary_planner],
    model=OpenAIChat(id="gpt-5.2"),
    instructions=[
        "You are a travel planning manager coordinating a team of specialists",
        "First, ask the Destination Researcher to gather info about the destination",
        "Then, ask the Hotel Finder to find accommodation options",
        "Finally, ask the Itinerary Planner to create a day-by-day schedule",
        "Combine all findings into a comprehensive, easy-to-read travel plan",
    ],
    show_members_responses=True,
    markdown=True,
)

# Wrap team execution in @observe to create a single grouped trace
@observe(name="Travel Planning Pipeline")
def plan_trip(query: str):
    """Run the travel planning team and return the response."""
    response = travel_team.run(query)

    # Update the current trace with input/output
    langfuse.update_current_trace(
        input=query,
        output=response.content if response else None,
    )
    return response


# Use the team
if __name__ == "__main__":
    query = "Plan a 5-day trip to Kyoto, Japan for a solo traveler interested in temples, traditional culture, and local food. Budget is mid-range."
    result = plan_trip(query)
    print(result.content if result else "No response")
    langfuse.flush()  # Ensure traces are sent before script exits