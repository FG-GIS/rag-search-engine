import json
import os
from time import sleep
from typing import Any,TypedDict
from sentence_transformers import CrossEncoder
from .gemini_wrap import query_gemma
from .openrouter_wrap import query_openrouter_free

class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]

DEFAULT_SEARCH_LIMIT = 5


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT,"data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT,"data", "stopwords.txt")
GOLDEN_PATH = os.path.join(PROJECT_ROOT,"data", "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3

def load_movies() -> list[dict]:
    with open(DATA_PATH,"r") as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words() -> list[str]:
    with open(STOP_WORDS_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]
    #     data = f.read()
    # return data.splitlines()

def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document[:100],
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def query_with_retry(query: str, max_retries: int = 5 , wait: int = 3,api: str = "google"):
    query_call = query_gemma

    match api:
        case "openrouter":
            query_call = query_openrouter_free

    for attempt in range(max_retries):
        try:
            return query_call(query)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = wait * (attempt + 1)
                print(f"Server error, retrying in {wait_time} s... ({attempt + 1}/{max_retries})")
                sleep(wait_time)
            else:
                raise e
    return ""

def enhance_query_spelling(query: str) -> str:
    prompt: str = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
    return query_with_retry(prompt)

def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User query: "{query}"
                """
    return query_with_retry(prompt)

def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User query: "{query}"
                """
    return query_with_retry(prompt)

def rerank_results(query: str, results: list[dict]) -> list[dict]:
    out : list = []
    for r in results:
        doc = r["document"]
        prompt = f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {doc.get("title", "")} - {doc.get("document", "")}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Output ONLY the number in your response, no other text or explanation.

                    Score:"""
        r["re_ranking"] = query_with_retry(prompt,max_retries=10,wait=60)
        out.append(r)
    return sorted(out, key=lambda item: item["re_ranking"], reverse=True)

def batch_rerank_results(query: str, results: list[dict]) -> list[dict]:
    doc_list_str = ""
    for r in results:
        doc = r["document"]
        doc_list_str += f"id:{doc["id"]},title:{doc["title"]},description:{doc["description"]};\n\n"

    prompt = f"""Rank the movies listed below by relevance to the following search query.

                Query: "{query}"

                Movies:
                {doc_list_str}

                Return the movie IDs in order of relevance, best match first.

                Your response must be a raw JSON array of integers.
                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                Do not include any explanatory text.

                For example:
                [75, 12, 34, 2, 1]

                Ranking:"""

    rankings_str = query_with_retry(prompt,max_retries=5,wait=5)
    if rankings_str == None:
        rankings_str = ""

    rankings_list = json.loads(rankings_str)

    rankings_dict = {}

    for score,id in enumerate(rankings_list, start=1):
        rankings_dict[id] = score

    out = []
    for r in results:
        r["batch_re_ranking"] = rankings_dict[r["document"]["id"]]
        out.append(r)
    return sorted(out,key= lambda item: item["batch_re_ranking"])

def cross_encoding(pairs: list):
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    return cross_encoder.predict(pairs)

def evaluate_results(query: str, formatted_results: list[str]) -> list[int]:
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{"".join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    scores_str = query_with_retry(prompt)
    if scores_str is None:
        scores_str = ""
    return json.loads(scores_str)

def simple_query(query: str, results: list, type: str) -> str:
    docs = ""
    for e in results:
        docs += e + "\n"

    start_prompt = ""
    query_addr = "Query"
    docs_addr = "Documents"
    ending = "Answer"

    match type:
        case "rag":
            start_prompt = """You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.\n"""

        case "summary":
            start_prompt = """Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.\n"""
            docs_addr = "Search results"
            ending = "Provide a comprehensive 3–4 sentence answer that combines information from multiple sources"

        case "citations":
            start_prompt = """Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.\n
"""
            ending = """Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer"""

        case "question":
            start_prompt = "Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.\n"
            query_addr = "Question"
            ending = """Instructions:
- Answer questions directly, concisely and thoroughly
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer"""

    prompt = f"""{start_prompt}
{query_addr}: {query}

{docs_addr}:
{docs}

{ending}:"""
    response = query_with_retry(prompt)
    if response is None:
        response = ""
    return response

