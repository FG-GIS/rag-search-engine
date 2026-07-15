import mimetypes
import argparse
import base64
from lib.openrouter_wrap import query_openrouter_free_full

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    parser.add_argument("--query", type=str, help="Query to rewrite.", required=True)
    parser.add_argument("--image", type=str, help="Image source", required=True)

    args = parser.parse_args()

    query = args.query
    image = args.image

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    # print(f"DEBUG -> mime = {mime}")

    with open(image, 'rb') as f:
        image_blob = f.read()

    data_url = f"data:{mime};base64,{base64.b64encode(image_blob).decode()}"

    # print(f"DEBUG -> data_url = {data_url}")

    sys_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    msgs = [
        {"type": "text", "text": sys_prompt.strip()},
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": query.strip()},
    ]
    response = query_openrouter_free_full(msgs)
    content = response.choices[0].message.content
    if content is None:
        content = ""
    print(f"Rewritten query: {content.strip()}")
    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")




if __name__ == "__main__":
    main()
