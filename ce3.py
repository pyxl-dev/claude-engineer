# ce3.py
import anthropic
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from typing import List, Dict, Any
import importlib
import inspect
import pkgutil
import os
import json
import sys
import logging
import asyncio

from config import Config
from tools.base import BaseTool
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from prompts.system_prompts import SystemPrompts

# Configure logging to only show ERROR level and above
logging.basicConfig(
    level=logging.ERROR,
    format='%(levelname)s: %(message)s'
)

class Assistant:
    """
    The Assistant class manages:
    - Loading of tools from a specified directory.
    - Interaction with the Anthropic or OpenRouter API (message completion).
    - Handling user commands such as 'refresh' and 'reset'.
    - Token usage tracking and display.
    - Tool execution upon request from model responses.
    """

    def __init__(self):
        self.provider = Config.PROVIDER
        
        # Initialize clients based on provider
        if self.provider == 'anthropic':
            if not Config.ANTHROPIC_API_KEY:
                raise ValueError("No ANTHROPIC_API_KEY found in environment variables")
            self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        elif self.provider == 'openrouter':
            if not Config.OPENROUTER_API_KEY:
                raise ValueError("No OPENROUTER_API_KEY found in environment variables")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=Config.OPENROUTER_API_KEY,
            )
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

        self.conversation_history: List[Dict[str, Any]] = []
        self.console = Console()
        self.thinking_enabled = getattr(Config, 'ENABLE_THINKING', False)
        self.temperature = getattr(Config, 'DEFAULT_TEMPERATURE', 0.7)
        self.total_tokens_used = 0
        self.tools = self._load_tools()

    def _execute_uv_install(self, package_name: str) -> bool:
        """
        Execute the uvpackagemanager tool directly to install the missing package.
        Returns True if installation seems successful (no errors in output), otherwise False.
        """
        class ToolUseMock:
            name = "uvpackagemanager"
            input = {
                "command": "install",
                "packages": [package_name]
            }

        result = self._execute_tool(ToolUseMock())
        if "Error" not in result and "failed" not in result.lower():
            self.console.print("[green]The package was installed successfully.[/green]")
            return True
        else:
            self.console.print(f"[red]Failed to install {package_name}. Output:[/red] {result}")
            return False

    def _load_tools(self) -> List[Dict[str, Any]]:
        """
        Dynamically load all tool classes from the tools directory.
        If a dependency is missing, prompt the user to install it via uvpackagemanager.
        
        Returns:
            A list of tools (dicts) containing their 'name', 'description', and 'input_schema'.
        """
        tools = []
        tools_path = getattr(Config, 'TOOLS_DIR', None)

        if tools_path is None:
            self.console.print("[red]TOOLS_DIR not set in Config[/red]")
            return tools

        # Clear cached tool modules for fresh import
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('tools.') and module_name != 'tools.base':
                del sys.modules[module_name]

        try:
            for module_info in pkgutil.iter_modules([str(tools_path)]):
                if module_info.name == 'base':
                    continue

                # Attempt loading the tool module
                try:
                    module = importlib.import_module(f'tools.{module_info.name}')
                    self._extract_tools_from_module(module, tools)
                except ImportError as e:
                    # Handle missing dependencies
                    missing_module = self._parse_missing_dependency(str(e))
                    self.console.print(f"\n[yellow]Missing dependency:[/yellow] {missing_module} for tool {module_info.name}")
                    user_response = input(f"Would you like to install {missing_module}? (y/n): ").lower()

                    if user_response == 'y':
                        success = self._execute_uv_install(missing_module)
                        if success:
                            # Retry loading the module after installation
                            try:
                                module = importlib.import_module(f'tools.{module_info.name}')
                                self._extract_tools_from_module(module, tools)
                            except Exception as retry_err:
                                self.console.print(f"[red]Failed to load tool after installation: {str(retry_err)}[/red]")
                        else:
                            self.console.print(f"[red]Installation of {missing_module} failed. Skipping this tool.[/red]")
                    else:
                        self.console.print(f"[yellow]Skipping tool {module_info.name} due to missing dependency[/yellow]")
                except Exception as mod_err:
                    self.console.print(f"[red]Error loading module {module_info.name}:[/red] {str(mod_err)}")
        except Exception as overall_err:
            self.console.print(f"[red]Error in tool loading process:[/red] {str(overall_err)}")

        return tools

    def _parse_missing_dependency(self, error_str: str) -> str:
        """
        Parse the missing dependency name from an ImportError string.
        """
        if "No module named" in error_str:
            parts = error_str.split("No module named")
            missing_module = parts[-1].strip(" '\"")
        else:
            missing_module = error_str
        return missing_module

    def _extract_tools_from_module(self, module, tools: List[Dict[str, Any]]) -> None:
        """
        Given a tool module, find and instantiate all tool classes (subclasses of BaseTool).
        Append them to the 'tools' list.
        """
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and issubclass(obj, BaseTool) and obj != BaseTool):
                try:
                    tool_instance = obj()
                    tools.append({
                        "name": tool_instance.name,
                        "description": tool_instance.description,
                        "input_schema": tool_instance.input_schema
                    })
                    self.console.print(f"[green]Loaded tool:[/green] {tool_instance.name}")
                except Exception as tool_init_err:
                    self.console.print(f"[red]Error initializing tool {name}:[/red] {str(tool_init_err)}")

    def refresh_tools(self):
        """
        Refresh the list of tools and show newly discovered tools.
        """
        current_tool_names = {tool['name'] for tool in self.tools}
        self.tools = self._load_tools()
        new_tool_names = {tool['name'] for tool in self.tools}
        new_tools = new_tool_names - current_tool_names

        if new_tools:
            self.console.print("\n")
            for tool_name in new_tools:
                tool_info = next((t for t in self.tools if t['name'] == tool_name), None)
                if tool_info:
                    description_lines = tool_info['description'].strip().split('\n')
                    formatted_description = '\n    '.join(line.strip() for line in description_lines)
                    self.console.print(f"[bold green]NEW[/bold green] 🔧 [cyan]{tool_name}[/cyan]:\n    {formatted_description}")
        else:
            self.console.print("\n[yellow]No new tools found[/yellow]")

    def display_available_tools(self):
        """
        Print a list of currently loaded tools.
        """
        self.console.print("\n[bold cyan]Available tools:[/bold cyan]")
        tool_names = [tool['name'] for tool in self.tools]
        if tool_names:
            formatted_tools = ", ".join([f"🔧 [cyan]{name}[/cyan]" for name in tool_names])
        else:
            formatted_tools = "No tools available."
        self.console.print(formatted_tools)
        self.console.print("---")

    def _clean_data_for_display(self, data):
        """
        Helper method to clean data for display by handling various data types
        and removing/replacing large content like base64 strings.
        """
        if isinstance(data, str):
            # Escape Rich markup characters
            data = data.replace("[", "\\[").replace("]", "\\]")
            # Truncate long strings
            if len(data) > 1000:
                return data[:1000] + "... [truncated]"
            return data
        elif isinstance(data, (dict, list)):
            try:
                # Try to parse as JSON if it's a string representation
                if isinstance(data, str):
                    data = json.loads(data)
                return self._clean_parsed_data(data)
            except json.JSONDecodeError:
                return data
        return str(data)

    def _clean_parsed_data(self, data):
        """
        Recursively clean parsed JSON/dict data, handling nested structures
        and replacing large data with placeholders.
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Handle image data in various formats
                if key in ['data', 'image', 'source'] and isinstance(value, str):
                    if len(value) > 1000 and (';base64,' in value or value.startswith('data:')):
                        cleaned[key] = "[base64 data omitted]"
                    else:
                        cleaned[key] = value
                else:
                    cleaned[key] = self._clean_parsed_data(value)
            return cleaned
        elif isinstance(data, list):
            return [self._clean_parsed_data(item) for item in data]
        elif isinstance(data, str) and len(data) > 1000 and ';base64,' in data:
            return "[base64 data omitted]"
        return data

    def _display_tool_usage(self, tool_name: str, input_data: Dict, result: str):
        """
        If SHOW_TOOL_USAGE is enabled, display the input and result of a tool execution.
        Handles special cases like image data and large outputs for cleaner display.
        """
        if not getattr(Config, 'SHOW_TOOL_USAGE', True):
            return

        try:
            # Clean and format the input data
            cleaned_input = self._clean_data_for_display(input_data)
            input_str = json.dumps(cleaned_input, indent=2) if isinstance(cleaned_input, (dict, list)) else str(cleaned_input)
            
            # Clean and format the result
            cleaned_result = self._clean_data_for_display(result)
            result_str = json.dumps(cleaned_result, indent=2) if isinstance(cleaned_result, (dict, list)) else str(cleaned_result)

            # Create a formatted display string with escaped markup
            display_str = (
                f"[bold blue]Tool:[/bold blue] {tool_name}\n"
                f"[bold green]Input:[/bold green]\n{input_str}\n"
                f"[bold yellow]Result:[/bold yellow]\n{result_str}"
            )

            # Display the formatted string
            self.console.print(Panel(display_str))

        except Exception as e:
            logging.error(f"Error displaying tool usage: {str(e)}")
            # Fallback to plain print if Rich formatting fails
            print(f"\nTool: {tool_name}")
            print(f"Input: {input_data}")
            print(f"Result: {result}\n")

    def _execute_tool(self, tool_use):
        """
        Given a tool usage request (with tool name and inputs),
        dynamically load and execute the corresponding tool.
        """
        tool_name = tool_use.name
        tool_input = tool_use.input or {}
        tool_result = None

        try:
            module = importlib.import_module(f'tools.{tool_name}')
            tool_instance = self._find_tool_instance_in_module(module, tool_name)

            if not tool_instance:
                tool_result = f"Tool not found: {tool_name}"
            else:
                # Execute the tool with the provided input
                try:
                    result = tool_instance.execute(**tool_input)
                    # Keep structured data intact
                    tool_result = result
                except Exception as exec_err:
                    tool_result = f"Error executing tool '{tool_name}': {str(exec_err)}"
        except ImportError:
            tool_result = f"Failed to import tool: {tool_name}"
        except Exception as e:
            tool_result = f"Error executing tool: {str(e)}"

        # Display tool usage with proper handling of structured data
        self._display_tool_usage(tool_name, tool_input, 
            json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result)
        return tool_result

    def _find_tool_instance_in_module(self, module, tool_name: str):
        """
        Search a given module for a tool class matching tool_name and return an instance of it.
        """
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and issubclass(obj, BaseTool) and obj != BaseTool):
                candidate_tool = obj()
                if candidate_tool.name == tool_name:
                    return candidate_tool
        return None

    async def get_completion(self, messages: List[Dict[str, str]], stream: bool = True) -> str:
        """Get a completion from the selected provider."""
        try:
            if self.provider == 'anthropic':
                return await self._get_anthropic_completion(messages, stream)
            else:
                return await self._get_openrouter_completion(messages, stream)
        except Exception as e:
            self.console.print(f"[red]Error getting completion: {str(e)}[/red]")
            return ""

    async def _get_openrouter_completion(self, messages: List[Dict[str, str]], stream: bool = True) -> str:
        """Get a completion from OpenRouter."""
        try:
            extra_headers = {}
            if Config.OPENROUTER_SITE_URL:
                extra_headers["HTTP-Referer"] = Config.OPENROUTER_SITE_URL
            if Config.OPENROUTER_APP_NAME:
                extra_headers["X-Title"] = Config.OPENROUTER_APP_NAME

            # Format messages for OpenRouter (OpenAI format)
            formatted_messages = []
            
            # Add system prompt first
            formatted_messages.append({
                "role": "system",
                "content": f"{SystemPrompts.DEFAULT}\n\n{SystemPrompts.TOOL_USAGE}"
            })
            
            # Add conversation messages
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    content_text = ""
                    for content_item in msg["content"]:
                        if isinstance(content_item, dict):
                            if content_item.get("type") == "text":
                                content_text += content_item.get("text", "") + "\n"
                            elif content_item.get("type") == "tool_result":
                                content_text += f"Tool Result: {content_item.get('content', '')}\n"
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": content_text.strip()
                    })
                else:
                    formatted_messages.append({
                        "role": msg.get("role", "user"),
                        "content": str(msg.get("content", ""))
                    })

            # Start with the configured model
            models_to_try = [Config.OPENROUTER_MODEL] + Config.OPENROUTER_FALLBACK_MODELS
            
            for model in models_to_try:
                self.console.print(f"[yellow]Trying model: {model}[/yellow]")
                
                try:
                    completion = self.client.chat.completions.create(
                        extra_headers=extra_headers,
                        model=model,
                        messages=formatted_messages,
                        temperature=self.temperature,
                        max_tokens=Config.MAX_TOKENS,
                        top_p=0.9,
                        presence_penalty=0.6,
                        frequency_penalty=0.6
                    )
                    
                    self.console.print(f"[green]Response received from {model}[/green]")
                    
                    if hasattr(completion, 'choices') and completion.choices and len(completion.choices) > 0:
                        content = completion.choices[0].message.content
                        if content:
                            return content
                    
                    # If we get here, the response wasn't valid
                    self.console.print(f"[yellow]Invalid response from {model}, trying next model...[/yellow]")
                    
                except Exception as model_error:
                    error_message = str(model_error)
                    if "maintenance" in error_message.lower():
                        self.console.print(f"[yellow]{model} is under maintenance, trying next model...[/yellow]")
                    elif "503" in error_message:
                        self.console.print(f"[yellow]{model} is temporarily unavailable, trying next model...[/yellow]")
                    else:
                        self.console.print(f"[red]Error with {model}: {error_message}[/red]")
                    continue
            
            # If we've exhausted all models
            return "I apologize, but I'm having trouble accessing the language models at the moment. Please try again later."
            
        except Exception as e:
            self.console.print(f"[red]Error in OpenRouter completion: {str(e)}[/red]")
            return f"Error: {str(e)}"

    async def _get_anthropic_completion(self, messages: List[Dict[str, str]], stream: bool = True) -> str:
        """Get a completion from Anthropic."""
        try:
            response = await self.client.messages.create(
                model=Config.ANTHROPIC_MODEL,
                messages=messages,
                temperature=self.temperature,
                stream=stream,
                max_tokens=Config.MAX_TOKENS
            )
            
            if hasattr(response, 'content') and isinstance(response.content, list):
                return response.content[0].text
            return ""
            
        except Exception as e:
            self.console.print(f"[red]Error in Anthropic completion: {str(e)}[/red]")
            return ""

    async def chat(self, user_input):
        """
        Process a chat message from the user.
        user_input can be either a string (text-only) or a list (multimodal message)
        """
        # Handle special commands only for text-only messages
        if isinstance(user_input, str):
            if user_input.lower() == 'refresh':
                self.refresh_tools()
                return "Tools refreshed successfully!"
            elif user_input.lower() == 'reset':
                self.reset()
                return "Conversation reset!"
            elif user_input.lower() == 'quit':
                return "Goodbye!"

        try:
            # Add user message to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_input  # This can be either string or list
            })

            # Show thinking indicator if enabled
            if self.thinking_enabled:
                with Live(Spinner('dots', text='Thinking...', style="cyan"), 
                         refresh_per_second=10, transient=True):
                    response = await self.get_completion(self.conversation_history)
            else:
                response = await self.get_completion(self.conversation_history)

            return response

        except Exception as e:
            logging.error(f"Error in chat: {str(e)}")
            return f"Error: {str(e)}"

    def reset(self):
        """
        Reset the assistant's memory and token usage.
        """
        self.conversation_history = []
        self.total_tokens_used = 0
        self.console.print("\n[bold green]🔄 Assistant memory has been reset![/bold green]")

        welcome_text = """
# Claude Engineer v3. A self-improving assistant framework with tool creation

Type 'refresh' to reload available tools
Type 'reset' to clear conversation history
Type 'quit' to exit

Available tools:
"""
        self.console.print(Markdown(welcome_text))
        self.display_available_tools()


def main():
    """
    Entry point for the assistant CLI loop.
    Provides a prompt for user input and handles 'quit' and 'reset' commands.
    """
    console = Console()
    style = Style.from_dict({'prompt': 'orange'})

    try:
        assistant = Assistant()
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("Please ensure ANTHROPIC_API_KEY is set correctly.")
        return

    welcome_text = """
# Claude Engineer v3. A self-improving assistant framework with tool creation

Type 'refresh' to reload available tools
Type 'reset' to clear conversation history
Type 'quit' to exit

Available tools:
"""
    console.print(Markdown(welcome_text))
    assistant.display_available_tools()

    while True:
        try:
            user_input = prompt("You: ", style=style).strip()

            if user_input.lower() == 'quit':
                console.print("\n[bold blue]👋 Goodbye![/bold blue]")
                break
            elif user_input.lower() == 'reset':
                assistant.reset()
                continue

            response = asyncio.run(assistant.chat(user_input))
            console.print("\n[bold purple]Claude Engineer:[/bold purple]")
            if isinstance(response, str):
                safe_response = response.replace('[', '\\[').replace(']', '\\]')
                console.print(safe_response)
            else:
                console.print(str(response))

        except KeyboardInterrupt:
            continue
        except EOFError:
            break


if __name__ == "__main__":
    main()