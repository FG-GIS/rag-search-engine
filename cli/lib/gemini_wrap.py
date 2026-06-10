import os
from dotenv import load_dotenv
from google import genai

load_dotenv("gemini_key.env")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"

def query_gemma(query: str) -> str:
    content = client.models.generate_content(contents=query,model=model)
    if content.text is None:
        return ""
    return content.text

