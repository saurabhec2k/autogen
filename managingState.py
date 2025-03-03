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

assistant_agent = AssistantAgent(
    name="assistant_agent",
    system_message="You are a helpful assistant",
    model_client=model_client
)
new_assistant_agent = AssistantAgent(
    name="assistant_agent",
    system_message="You are a helpful assistant",
     model_client=model_client
)
# Run the conversation and stream to the console
async def main() -> None:
    
    # Iterate over the stream returned by run_stream
    response = await assistant_agent.on_messages(
    [TextMessage(content="Write a 3 line poem on lake tangayika", source="user")], CancellationToken()
    )
    print(response.chat_message.content)
    agent_state = await assistant_agent.save_state()
    print(agent_state)
    await new_assistant_agent.load_state(agent_state)
      # Iterate over the stream returned by run_stream
    response = await new_assistant_agent.on_messages(
    [TextMessage(content="What was the last line of the previous poem you wrote", source="user")], CancellationToken()
    )
    print(response.chat_message.content)

# NOTE: if running this inside a Python script, you'll need to use asyncio.run(main())
if __name__ == "__main__":
    asyncio.run(main())  # This ensures the event loop runs correctly.