import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stop_words
from nltk.stem import PorterStemmer

def search_command(query:str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    for movie in movies:
        prep_query = process_text(query)
        prep_title = process_text(movie["title"])
        if has_matching_token(prep_query,prep_title):
            results.append(movie)
            if len(results)>= limit:
                break
            
    return  results

def process_text(text:str) -> list[str]:
    return stem_words(remove_stop_words(tokenize_text(preprocess_text(text))))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

def tokenize_text(text:str) -> list[str]:
    output = text.split(" ")
    output = [item for item in output if item.strip()]
    return output

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for q_token in query_tokens:
        for t_token in title_tokens:
            if q_token in t_token:
                return True
    return False

def remove_stop_words(t_list:list[str]) -> list[str]:
    stop_words = load_stop_words()
    for w in stop_words:
        if w in t_list:
            t_list.remove(w)
    return t_list

def stem_words(list: list[str]) -> list[str]:
    result = []
    stemmer = PorterStemmer()
    for item in list:
        result.append(stemmer.stem(item))
    return result
