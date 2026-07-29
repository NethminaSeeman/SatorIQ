import os
from langchain_groq import ChatGroq

class GroqClient:
    """
    Client wrapper for initializing Groq models for LangChain/LangGraph usage.
    Used for Router, Summary, and Reflection agents.
    """
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.model_name = model_name
        # Assumes GROQ_API_KEY is available in the environment
        if not os.environ.get("GROQ_API_KEY"):
            print("Warning: GROQ_API_KEY environment variable not set.")
            
    def get_llm(self):
        """
        Returns the LangChain ChatGroq instance.
        """
        return ChatGroq(
            model_name=self.model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
