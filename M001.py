import asyncio
import sys
from typing import List, Dict, AsyncGenerator, Optional
from dataclasses import dataclass, field
from ollama import AsyncClient, ResponseError

# --- Configuration ---
@dataclass
class AgentConfig:
    """Configuration settings for the Ollama Agent."""
    model_name: str = "llama3.2"  # Change to your installed model (e.g., mistral, llama3)
    system_prompt: str = "You are a helpful, precise AI assistant. Answer concisely."
    temperature: float = 0.7

# --- The Engine ---
class OllamaEngine:
    """
    A robust wrapper around the Ollama AsyncClient.
    Manages conversation history and streaming responses.
    """
    def __init__(self, config: AgentConfig):
        self.client = AsyncClient()
        self.config = config
        self.history: List[Dict[str, str]] = []
        
        # Initialize history with system prompt
        if self.config.system_prompt:
            self.history.append({"role": "system", "content": self.config.system_prompt})

    async def stream_response(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Sends input to Ollama and yields chunks of text as they generated.
        Updates internal history automatically.
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_input})
        
        full_response_buffer = ""

        try:
            # Create the chat completion stream
            async for part in await self.client.chat(
                model=self.config.model_name,
                messages=self.history,
                stream=True,
                options={"temperature": self.config.temperature}
            ):
                token = part['message']['content']
                full_response_buffer += token
                yield token

            # Once stream finishes, append full assistant response to history
            self.history.append({"role": "assistant", "content": full_response_buffer})

        except ResponseError as e:
            yield f"\n[!] Ollama API Error: {e.error}"
        except Exception as e:
            yield f"\n[!] Unexpected Error: {str(e)}"

    def clear_memory(self):
        """Resets the conversation history."""
        self.history = [x for x in self.history if x["role"] == "system"]
        print("\n--- Memory Cleared ---\n")

# --- Main Application Loop ---
async def main():
    # 1. Setup
    config = AgentConfig(model_name="llama3.2") # Ensure this model is pulled in Ollama
    agent = OllamaEngine(config)
    
    print(f"--- Advanced Ollama Client (Model: {config.model_name}) ---")
    print("Type 'exit' to quit or 'clear' to reset memory.\n")

    # 2. Event Loop
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break
            
            if user_input.lower() == "clear":
                agent.clear_memory()
                continue

            # 3. Output Handling (Streaming)
            print("Response: ", end="", flush=True)
            
            async for chunk in agent.stream_response(user_input):
                # Print directly to stdout without newline to create typing effect
                sys.stdout.write(chunk)
                sys.stdout.flush()
            
            print("\n")  # Newline after response completes

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            break

if __name__ == "__main__":
    # Run the async main loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass