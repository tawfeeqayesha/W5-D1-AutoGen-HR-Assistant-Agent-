import asyncio
import os
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize the model client
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Create AssistantAgent with the EXACT system message required
agent = AssistantAgent(
    name="hr_assistant", 
    model_client=model_client,
    system_message="You are an HR assistant."
)

async def main():
    # Clarify "5 lines" as 5 bullet points to eliminate ambiguity
    task = (
        "Write a job description for a Junior Python Developer. "
        "Your response must be exactly 5 lines long, formatted as 5 distinct bullet points."
    )
    
    # Stream output using Console
    await Console(agent.run_stream(task=task))
    
    # Always close the model client to prevent hanging
    await model_client.close()

if __name__ == "__main__":
    asyncio.run(main())