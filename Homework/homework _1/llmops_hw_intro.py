import requests
import tiktoken
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

def fetch_documents():
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)

    return documents

def create_elasticsearch_client():
    es_client = Elasticsearch('http://localhost:9200')
    es_client.info()
    return es_client

def create_index(es_client, index_name):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"}
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_settings)

def index_documents(es_client, index_name, documents):
    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)

def elastic_search(es_client, query, course='data-engineering-zoomcamp', size=3):
    search_query = {
        "size": size,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    result_docs = [hit['_source'] for hit in response['hits']['hits']]
    return result_docs, response['hits']['max_score']

def format_context(result_docs):
    context_template = """
    Q: {question}
    A: {text}
    """.strip()

    return "\n\n".join(context_template.format(**doc) for doc in result_docs)

def create_prompt(query, context):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    return prompt_template.format(question=query, context=context)

def count_tokens_tiktoken(text, model="gpt-4o"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    return len(tokens)

if __name__ == "__main__":
    documents = fetch_documents()
    es_client = create_elasticsearch_client()
    index_name = "course-questions"
    create_index(es_client, index_name)
    index_documents(es_client, index_name, documents)

    query = "How do I execute a command in a running docker container?"
    result_docs, max_score = elastic_search(es_client, query)
    print(max_score)

    result_docs, _ = elastic_search(es_client, query, course='machine-learning-zoomcamp')
    third_question = result_docs[2]['question'] if len(result_docs) > 2 else None
    print(third_question)

    context = format_context(result_docs)
    prompt = create_prompt(query, context)

    prompt_length = len(prompt)
    print(prompt_length)

    token_count = count_tokens_tiktoken(prompt)
    print(token_count)
