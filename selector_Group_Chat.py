import asyncio
import os
from dotenv import load_dotenv
from typing import Sequence
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import AgentEvent, ChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Load environment variables
load_dotenv()

# Create the token provider
token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")

API_KEY = os.getenv("API_KEY")
MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME")
MODEL_API_VERSION = os.getenv("MODEL_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")

# Define a model client
model_client = AzureOpenAIChatCompletionClient(
    azure_deployment=MODEL_DEPLOYMENT_NAME,
    model=MODEL_DEPLOYMENT_NAME,
    api_version=MODEL_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=API_KEY
)

# Note: This example uses mock tools instead of real APIs for demonstration purposes
def search_web_tool(query: str) -> str:
    if "2006-2007" in query:
        return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
        Udonis Haslem: 844 points
        Dwayne Wade: 1397 points
        James Posey: 550 points
        ...
        """
    elif "2007-2008" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2007-2008 is 214."
    elif "2008-2009" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2008-2009 is 398."
    return "No data found."


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100

planning_agent = AssistantAgent(
    "PlanningAgent",
    description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message="""\
    You are a planning agent.
    Your job is to break down complex tasks into smaller, manageable subtasks.
    Your team members are:
        WebSearchAgent: Searches for information
        DataAnalystAgent: Performs calculations

    You only plan and delegate tasks - you do not execute them yourself.

    When assigning tasks, use this format:
    1. <agent> : <task>

    After all tasks are complete, summarize the findings and end with "TERMINATE".
    """
)

web_search_agent = AssistantAgent(
    "WebSearchAgent",
    description="An agent for searching information on the web.",
    tools=[search_web_tool],
    model_client=model_client,
    reflect_on_tool_use=True,
    system_message="""\
    You are a web search agent.
    Your only tool is search_tool - use it to find information.
    You make only one search call at a time.
    Once you have the results, you never do calculations based on them.
    """
)

data_analyst_agent = AssistantAgent(
    "DataAnalystAgent",
    description="An agent for performing calculations.",
    model_client=model_client,
    tools=[percentage_change_tool],
    reflect_on_tool_use=True,
    system_message="""\
    You are a data analyst.
    Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
    If you have not seen the data, ask for it.
    """
)

text_mention_termination = TextMentionTermination("TERMINATE")
max_messages_termination = MaxMessageTermination(max_messages=25)
termination = text_mention_termination | max_messages_termination

# SelectorGroupChat uses a model to select the next speaker based on the 
# conversation context. We will use a custom selector prompt to properly align with the workflow.

selector_prompt = """\
Select an agent to perform task.

{roles}

Current conversation context:
{history}

Read the above conversation, then select an agent from {participants} to perform the next task.
Make sure the planner agent has assigned tasks before other agents start working.
Only select one agent.
"""

# team = SelectorGroupChat(
#     [planning_agent, web_search_agent, data_analyst_agent],
#     model_client=model_client,
#     termination_condition=termination,
#     selector_prompt=selector_prompt,
#     allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
# )

# Custom Selector Function
# Often times we want better control over the selection process. 
# To this end, we can set the selector_func argument with a custom selector function to override the default 
# model-based selection. For instance, we want the Planning Agent to speak immediately after any specialized agent to check the progress.

def selector_func(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
    if messages[-1].source != planning_agent.name:
        return planning_agent.name
    return None

team = SelectorGroupChat(
    [planning_agent, web_search_agent, data_analyst_agent],
    model_client=model_client,
    termination_condition=termination,
    selector_prompt=selector_prompt,
    allow_repeated_speaker=True,
    selector_func=selector_func,
)

# User Feedback
# We can add UserProxyAgent to the team to provide user feedback during a run. 
# See Human-in-the-Loop for more details about UserProxyAgent.

# user_proxy_agent = UserProxyAgent("UserProxyAgent", description="A proxy for the user to approve or disapprove tasks.")


# def selector_func_with_user_proxy(messages: Sequence[AgentEvent | ChatMessage]) -> str | None:
#     if messages[-1].source != planning_agent.name and messages[-1].source != user_proxy_agent.name:
#         # Planning agent should be the first to engage when given a new task, or check progress.
#         return planning_agent.name
#     if messages[-1].source == planning_agent.name:
#         if messages[-2].source == user_proxy_agent.name and "APPROVE" in messages[-1].content.upper():  # type: ignore
#             # User has approved the plan, proceed to the next agent.
#             return None
#         # Use the user proxy agent to get the user's approval to proceed.
#         return user_proxy_agent.name
#     if messages[-1].source == user_proxy_agent.name:
#         # If the user does not approve, return to the planning agent.
#         if "APPROVE" not in messages[-1].content.upper():  # type: ignore
#             return planning_agent.name
#     return None

# team.reset()
# team = SelectorGroupChat(
#     [planning_agent, web_search_agent, data_analyst_agent, user_proxy_agent],
#     model_client=model_client,
#     termination_condition=termination,
#     selector_prompt=selector_prompt,
#     selector_func=selector_func_with_user_proxy,
#     allow_repeated_speaker=True,
# )

task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"

# The following is the main entry point for running the asynchronous logic
async def run_task():
    await Console(team.run_stream(task=task))
    
# Reset the previous team and run the chat again with the selector function.
    await team.reset()
   

# Ensure the event loop is running correctly with asyncio.run
asyncio.run(run_task())  # This ensures the event loop runs correctly.
