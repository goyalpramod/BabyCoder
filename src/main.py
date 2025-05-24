import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from models.agent import Agent

load_dotenv(find_dotenv())  

async def get_user_message():
    """Function to get user input from the console."""
    try:
        user_input = input("\033[94mYou\033[0m: ")
        return user_input, True
    except EOFError:
        return "", False

async def main():
    """Main entry point for the application."""
    # Initialize the OpenAI client
    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
    # Create tools list (we'll add these later)
    tools = []
    
    # Create and run the agent
    agent = Agent(
        client=client,
        get_user_message=get_user_message,
        tools=tools
    )
    
    try:
        await agent.run()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())