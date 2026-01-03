import asyncio
import sys
from typing import (
    List,
    Dict,
    AsyncGenerator,
    Optional
)
import json
import datetime
from dataclasses import dataclass, field
from ollama import AsyncClient, ResponseError
import logging
from pathlib import Path


logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# --- Configuration ---
@dataclass
class AgentConfig:
    """Configuration settings for the Ollama Agent."""
    model_name: str = "llama3.2"  # Change to your installed model (e.g., mistral, llama3)
    system_prompt: str = "You are a helpful, precise AI assistant. Answer concisely."
    temperature: float = 0.7
    base_url = "http://localhost:11111" #change it according to the user's llm configuration and compatibility
    top_p: float = 0.9

    #are to be set for more feature configurations
    top_k: int = 40
    num_predict: Optional[int] = None  # Max tokens to generate
    seed: Optional[int] = None         # For reproducibility
    keep_alive: str = "5m"             # Time for how long to keep model loaded
    context_length: int = 4096         # For history management or user's preference
    history_file: Optional[Path] = None


# --- The Engine ---
class OllamaEngine:
    """
    A robust wrapper around the Ollama AsyncClient.
    Manages conversation history and streaming responses.
    """
    def __init__(self, config: AgentConfig):
        self.client = AsyncClient(host=config.base_url)
        self.config = config
        self.history: List[Dict[str, str]] = []
        self.total_tokens_used: int = 0

        # init the system prompt
        if self.config.system_prompt:
            self.history.append({
                "role": "system",
                "content": self.config.system_prompt
            })

        # loading the history file if it is saved
        if self.config.history_file and self.config.history_file.exists():
            self.load_history()

    async def check_model_availability(self) -> bool:
        """Check if the configured model is available."""
        try:
            models = await self.client.list()
            available_models = [model['name'] for model in models['models']]
            if self.config.model_name not in available_models:
                logger.warning(f"Model {self.config.model_name} not found. Available: {available_models}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking models: {e}")
            return False

    async def stream_response(self,user_input: str,stream: bool = True) -> AsyncGenerator[str, None]:
        """
        Sends input to Ollama and yields chunks of text as they generated.
        Updates internal history automatically.
        """
        # appending history
        self.history.append({"role": "user", "content": user_input})

        # manage context length
        self._manage_context_length()

        full_response_buffer = ""

        try:
            #optional arguments
            options = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
            }

            if self.config.num_predict:
                options["num_predict"] = self.config.num_predict
            if self.config.seed:
                options["seed"] = self.config.seed

            # chat completion success message
            if stream:
                response = await self.client.chat(
                    model=self.config.model_name,
                    messages=self.history,
                    stream=True,
                    options=options
                )

                async for part in response:
                    token = part['message']['content']
                    full_response_buffer += token
                    yield token

                    self.total_tokens_used += 1
            else:
                # non-streaming response
                response = await self.client.chat(
                    model=self.config.model_name,
                    messages=self.history,
                    stream=False,
                    options=options
                )
                full_response_buffer = response['message']['content']
                self.total_tokens_used += len(full_response_buffer.split())
                yield full_response_buffer

            # after stream finishes, append full response to history
            self.history.append({
                "role": "assistant",
                "content": full_response_buffer
            })

            # saving history
            if self.config.history_file:
                self.save_history()

        except ResponseError as e:
            error_msg = f"\n[!] Ollama API Error: {e.error}"
            logger.error(error_msg)
            yield error_msg
        except asyncio.CancelledError:
            logger.info("Stream cancelled by user")
            yield "\n\n[Response interrupted]"
            raise
        except Exception as e:
            error_msg = f"\n[!] Unexpected Error: {str(e)}"
            logger.exception("Unexpected error in stream_response")
            yield error_msg

    def _manage_context_length(self):
        """Trim history if it exceeds context length (simple implementation)."""
        if len(self.history) > self.config.context_length:
            system_msg = [msg for msg in self.history if msg["role"] == "system"]
            recent_messages = self.history[-50:]  # Keep latest 50 exchanges
            self.history = system_msg + recent_messages
            logger.info("Trimmed conversation history")

    def clear_memory(self):
        """Resets the conversation history while keeping system prompt."""
        system_messages = [msg for msg in self.history if msg["role"] == "system"]
        self.history = system_messages
        logger.info("Memory cleared")
        return True

    def save_history(self):
        """Save conversation history to file."""
        if not self.config.history_file:
            return

        try:
            history_data = {
                "timestamp": datetime.now().isoformat(),
                "model": self.config.model_name,
                "history": self.history,
                "total_tokens": self.total_tokens_used
            }

            with open(self.config.history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            logger.info(f"History saved to {self.config.history_file}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def load_history(self):
        """Load conversation history from file."""
        try:
            with open(self.config.history_file, 'r') as f:
                data = json.load(f)
                self.history = data.get('history', [])
                self.total_tokens_used = data.get('total_tokens', 0)
            logger.info(f"History loaded from {self.config.history_file}")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current conversation."""
        user_messages = sum(1 for msg in self.history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in self.history if msg["role"] == "assistant")

        return {
            "total_messages": len(self.history) - 1,  # except system
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "total_tokens_used": self.total_tokens_used,
            "current_context_length": len(self.history)
        }

# --- Main Application Loop ---
async def main():
    # congfig setup
    config = AgentConfig(
        model_name="llama3.2",
        history_file=Path("./conversation_history.json"),
        temperature=0.7,
        base_url="http://localhost:11111"
    )

    agent = OllamaEngine(config)

    print(f"--- Advanced Ollama Client (Model: {config.model_name}) ---")
    print("Commands: 'exit', 'quit', 'clear', 'stats', 'save', 'load', 'models'")
    print("Type '/help' for more commands\n")

    # check the guard rail of model  availability
    if not await agent.check_model_availability():
        print(f"\n Model '{config.model_name}' may not be available.")
        print("Available models:")
        try:
            models = await agent.client.list()
            for model in models['models']:
                print(f"  - {model['name']}")
        except:
            print("  (Could not fetch model list)")
        print("\nContinue anyway? [y/N]: ", end="")
        if input().lower() != 'y':
            return

    # event loop
    while True:
        try:
            user_input = input("\n>>> ").strip()

            if not user_input:
                continue

            # command kwargs handling
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            elif user_input.lower() == "clear":
                agent.clear_memory()
                print("Memory cleared.")
                continue

            elif user_input.lower() == "stats":
                stats = agent.get_conversation_stats()
                print(f"\nConversation Statistics:")
                print(f"  Total messages: {stats['total_messages']}")
                print(f"  User messages: {stats['user_messages']}")
                print(f"  Assistant messages: {stats['assistant_messages']}")
                print(f"  Estimated tokens used: {stats['total_tokens_used']}")
                continue

            elif user_input.lower() == "save":
                agent.save_history()
                continue

            elif user_input.lower() == "load":
                agent.load_history()
                print("History loaded.")
                continue

            elif user_input.lower() == "models":
                try:
                    models = await agent.client.list()
                    print("\nAvailable models:")
                    for model in models['models']:
                        size_gb = model['size'] / 1e9
                        print(f"  - {model['name']} ({size_gb:.1f} GB)")
                except Exception as e:
                    print(f"Error fetching models: {e}")
                continue

            elif user_input.lower() == "/help":
                print("\nAvailable commands:")
                print("  exit/quit     - Exit the program")
                print("  clear         - Clear conversation history")
                print("  stats         - Show conversation statistics")
                print("  save          - Save conversation history")
                print("  load          - Load conversation history")
                print("  models        - List available models")
                print("  /help         - Show this help message")
                continue

            # streaming with the cancellation support
            print("Assistant: ", end="", flush=True)

            try:
                async for chunk in agent.stream_response(user_input):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                print()

            except asyncio.CancelledError:
                print("\n[Response cancelled]")
                # remove the clutter or unfinished things from history
                if agent.history and agent.history[-1]["role"] == "assistant":
                    agent.history.pop()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'exit' to quit or continue chatting.")
            continue
        except EOFError:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\n[!] Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram terminated.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\nFatal error: {e}")
        sys.exit(1)