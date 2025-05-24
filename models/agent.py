import json
from typing import Callable, List, Awaitable, Tuple, Dict, Any
from pydantic import BaseModel, ConfigDict
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict
    function: Callable

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Agent(BaseModel):
    client: AsyncOpenAI
    get_user_message: Callable[[], Awaitable[Tuple[str, bool]]]
    tools: List[ToolDefinition]

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    async def run(self):
        """Run the agent, handling the conversation loop."""
        print("Chat with GPT (use 'ctrl-c' to quit)")
        
        conversation: List[ChatCompletionMessageParam] = []
        
        read_user_input = True
        while True:
            if read_user_input:
                # Get user input
                user_input, ok = await self.get_user_message()
                if not ok:
                    break
                    
                # Add user message to conversation
                conversation.append({"role": "user", "content": user_input})
            
            # Get AI response
            response = await self.run_inference(conversation)
            
            # Handle response content and tools
            message = response.choices[0].message
            conversation.append(message)
            
            # Check for tool calls
            if message.tool_calls:
                read_user_input = False
                # Process tool calls
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_id = tool_call.id
                    
                    # Find the tool and execute it
                    tool_result = await self.execute_tool(tool_id, tool_name, tool_call.function.arguments)
                    
                    # Add tool result to conversation
                    conversation.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_result,
                    })
                
                # Continue loop to get AI's response after tool execution
                continue
            else:
                read_user_input = True
                # Print the response
                if message.content:
                    print(f"\033[93mGPT\033[0m: {message.content}")
    
    async def run_inference(self, conversation):
        """Send the conversation to OpenAI and get a response."""
        # Convert tools to OpenAI format
        openai_tools = []
        for tool in self.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema
                }
            })
        
        # Prepare the API call parameters
        api_params = {
            "model": "gpt-4o-mini",
            "messages": conversation,
        }
        
        # Only add tools and tool_choice if we have tools
        if openai_tools:
            api_params["tools"] = openai_tools
            api_params["tool_choice"] = "auto"
        
        # Call the API
        response = await self.client.chat.completions.create(**api_params)
        
        return response
    
    async def execute_tool(self, tool_id: str, tool_name: str, arguments_json: str) -> str:
        """Execute a tool with the given arguments."""
        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            return "Tool not found"
        
        # Parse arguments
        arguments = json.loads(arguments_json)
        
        # Print tool execution info
        print(f"\033[92mtool\033[0m: {tool_name}({arguments_json})")
        
        try:
            # Execute the tool
            result = tool.function(arguments)
            return result
        except Exception as e:
            return f"Error executing tool: {str(e)}"