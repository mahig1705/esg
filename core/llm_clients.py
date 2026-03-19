import os
from openai import AsyncOpenAI
import google.generativeai as genai
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

# Cerebras — OpenAI-compatible, fastest inference
cerebras = AsyncOpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ.get("CEREBRAS_API_KEY", "dummy"),
)

# Groq — OpenAI-compatible, fast + reliable Llama/Gemma
groq_client = AsyncGroq(
    api_key=os.environ.get("GROQ_API_KEY", "dummy"),
)

# OpenRouter — OpenAI-compatible, free model marketplace
openrouter = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "dummy"),
    default_headers={
        "HTTP-Referer": os.getenv("PLATFORM_URL", "https://esglens.com"),
        "X-Title":      os.getenv("PLATFORM_NAME", "ESGLens"),
    }
)

# Gemini — direct Google AI Studio, multimodal
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "dummy"))

def get_gemini(model_name: str):
    return genai.GenerativeModel(model_name)
