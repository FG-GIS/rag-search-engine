import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv("gemini_key.env")
api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

messages : list[ChatCompletionMessageParam]=[
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]

response = client.chat.completions.create(model="openrouter/free",messages=messages)
if response is not None and response.usage is not None:
    print(f"""Response: {response.choices[0].message.content}

Prompt tokens: {response.usage.prompt_tokens}
Response tokens: {response.usage.completion_tokens}""")
