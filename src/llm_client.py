import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqLLM:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in your AWS Environment/Secrets.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        print(f"GroqLLM initialized with model: {self.model}")

    def generate(self, prompt: str) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Groq API Error: {e}")
            return ""