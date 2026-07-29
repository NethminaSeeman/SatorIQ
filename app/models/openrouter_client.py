import os
from langchain_openai import ChatOpenAI

class OpenRouterClient:
    """
    Client wrapper for generalizing OpenRouter models via LangChain's OpenAI integration.
    Used for the Analysis agent.
    """
    def __init__(self, model_name: str = "meta-llama/llama-3.1-8b-instruct:free"):
        self.model_name = model_name
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            print("Warning: OPENROUTER_API_KEY environment variable not set.")
            
    def get_llm(self):
        """
        Returns the LangChain LLM instance configured for OpenRouter.
        """
        return ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.2, # Slight temperature for analysis creativity
            default_headers={
                "HTTP-Referer": "http://localhost:8501", 
                "X-Title": "SatorIQ",
            }
        )
