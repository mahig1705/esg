import os
from openai import AsyncOpenAI
from google import genai as google_genai
from groq import AsyncGroq
from dotenv import load_dotenv

load_dotenv()

cerebras = AsyncOpenAI(
    base_url="https://api.cerebras.ai/v1",
    api_key=os.environ["CEREBRAS_API_KEY"],
)

groq_client = AsyncGroq(
    api_key=os.environ["GROQ_API_KEY"],
)

openrouter = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    default_headers={
        "HTTP-Referer": os.getenv("PLATFORM_URL", "https://esglens.com"),
        "X-Title":      os.getenv("PLATFORM_NAME", "ESGLens"),
    }
)

gemini_client = google_genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)
