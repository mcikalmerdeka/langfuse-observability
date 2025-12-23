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

# Define structured output schemas
class ResearchFindings(BaseModel):
    topic: str
    key_findings: str
    sources: str


class TrendAnalysis(BaseModel):
    topic: str
    current_trends: str
    market_impact: str


class TechResearchReport(BaseModel):
    topic: str
    executive_summary: str
    key_findings: str
    trend_analysis: str
    sources: str


# Create specialized agents
research_agent = Agent(
    id="research-agent",
    name="Research Agent",
    role="Expert at finding and gathering information from the web",
    description="You are a skilled researcher who finds accurate, up-to-date information on technology topics.",
    instructions=[
        "Search for the most recent and relevant information",
        "Focus on credible sources and verified data",
        "Summarize findings clearly and concisely",
    ],
    model=OpenAIChat(id="gpt-4.1-mini"),
    tools=[TavilyTools()],
    output_schema=ResearchFindings,
    markdown=True,
)

analyst_agent = Agent(
    id="analyst-agent",
    name="Trend Analyst Agent",
    role="Expert at analyzing technology trends and market impact",
    description="You are a technology analyst who identifies patterns, trends, and their implications.",
    instructions=[
        "Analyze the research findings for key trends",
        "Identify market implications and future directions",
        "Provide actionable insights",
    ],
    model=OpenAIChat(id="gpt-4.1-mini"),
    output_schema=TrendAnalysis,
    markdown=True,
)

# Create the team
tech_research_team = Team(
    name="Technology Research Team",
    members=[research_agent, analyst_agent],
    model=OpenAIChat(id="gpt-5-mini"),
    instructions=[
        "Coordinate with team members to provide comprehensive technology research",
        "First, delegate research tasks to gather information",
        "Then, delegate analysis tasks to identify trends and insights",
        "Combine findings into a cohesive final report",
    ],
    output_schema=TechResearchReport,
    show_members_responses=True,
    markdown=True,
    debug_mode=True,
)

# Wrap team execution in @observe to create a single grouped trace
@observe(name="Tech Research Team Pipeline")
def run_tech_research(query: str):
    """Run the tech research team and return the response."""
    response = tech_research_team.run(query)
    
    # Update the current trace with input/output
    langfuse.update_current_trace(
        input=query,
        output=response.content if response else None,
    )
    return response


# Use the team
if __name__ == "__main__":
    result = run_tech_research("What is currently trending AI news in IDE business of the year 2025?")
    print(result.content if result else "No response")
    langfuse.flush()  # Ensure traces are sent before script exits