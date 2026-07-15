import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv("gemini_key.env")
api_key = os.environ.get("OPENROUTER_API_KEY")
MODEL = "openrouter/free"

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

def query_openrouter_free(query: str) -> str:
    if query == "" or query is None:
        return ""
    messages : list[ChatCompletionMessageParam]=[
        {
            "role": "user",
            "content": query,
        }
    ]
    response = client.chat.completions.create(model=MODEL,messages=messages)
    str_resp = response.choices[0].message.content
    if str_resp is None:
        raise Exception("None response from OpenRouter.")
    return str_resp

def query_openrouter_free_full(query):
    if query == "" or query is None:
        raise Exception("None response from OpenRouter.")
    messages : list[ChatCompletionMessageParam]=[
        {
            "role": "user",
            "content": query,
        }
    ]
    response = client.chat.completions.create(model=MODEL,messages=messages)
    if response is None:
        raise Exception("None response from OpenRouter.")
    return response
