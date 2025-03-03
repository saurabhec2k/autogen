import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import json
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

# Define a team.
assistant_agent = AssistantAgent(
    name="assistant_agent",
    system_message="You are a helpful assistant",
    model_client=model_client
)
agent_team = RoundRobinGroupChat([assistant_agent], termination_condition=MaxMessageTermination(max_messages=2))
# Run the conversation and stream to the console
async def main() -> None:
    
    async for message in agent_team.run_stream(task="Write a beautiful poem 3-line about lake tangayika"):
        print(message)
    # Save the state of the agent team.
    team_state = await agent_team.save_state()
    with open("coding/team_state.json", "w") as f:
        json.dump(team_state, f)
    ## load state from disk
    with open("coding/team_state.json", "r") as f:
        team_state = json.load(f)
    new_agent_team = RoundRobinGroupChat([assistant_agent], termination_condition=MaxMessageTermination(max_messages=2))
    await new_agent_team.load_state(team_state)
    #await agent_team.load_state(team_state)
    # async for message in agent_team.run_stream(task="What was the last line of the poem you wrote?"):
    async for message in new_agent_team.run_stream(task="What was the last line of the poem you wrote?"):
        print(message)

# NOTE: if running this inside a Python script, you'll need to use asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())  # This ensures the event loop runs correctly.