import os
from dotenv import load_dotenv
from google import genai

load_dotenv("gemini_key.env")
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

client = genai.Client(api_key=api_key)

content = client.models.generate_content(contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",model="gemma-4-31b-it")
print(content.text)
if content.usage_metadata != None:
    print(f"Prompt tokens: {content.usage_metadata.prompt_token_count}")
    print(f"Response tokens: {content.usage_metadata.candidates_token_count}")
