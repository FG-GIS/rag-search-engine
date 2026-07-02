import os
from time import sleep
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv("gemini_key.env")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)
model = "gemma-4-31b-it"
# model = "gemini-2.5-flash-lite"

def query_gemma(query: str) -> str:
    content = client.models.generate_content(contents=query,model=model)
    if content.text is None:
        return ""
    return content.text

def query_with_retry(query: str, max_retries: int = 5 , wait: int = 3):
    for attempt in range(max_retries):
        try:
            return query_gemma(query)
        except errors.ServerError as e:
            if attempt < max_retries - 1:
                wait_time = wait * (attempt + 1)
                print(f"Server error, retrying in {wait_time} s... ({attempt + 1}/{max_retries})")
                sleep(wait_time)
            else:
                raise e
