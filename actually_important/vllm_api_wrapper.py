"""Wrapper for vLLM server using OpenAI-compatible API."""

from openai import OpenAI
import threading


class SimpleLLMWrapper:
    """Simple thread-safe wrapper for vLLM server via OpenAI API."""
    
    def __init__(self, api_base: str = "http://localhost:8000/v1", api_key: str = "dummy"):
        """
        Args:
            api_base: Base URL for vLLM server (default: http://localhost:8000/v1)
            api_key: API key (not used by vLLM, but required by OpenAI client)
        """
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key
        )
        self.lock = threading.Lock()
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """
        Generate text from a prompt. Thread-safe.
        Matches the signature of the original Model.generate() method.
        
        Args:
            prompt: The input prompt
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 512)
            
        Returns:
            Generated text as a single string
        """
        # Thread-safe generation
        with self.lock:
            response = self.client.completions.create(
                model="",  # vLLM doesn't care about model name when using completions
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        # Extract the generated text
        return response.choices[0].text


