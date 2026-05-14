
import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

EMBEDDDING_MODEL='text-embedding-3-small'
DIM=1536


collection_name="test_rag"
client = chromadb.HttpClient( host="localhost", port=8200) # Connect to Docker
print(client.heartbeat())


def init_chroma_open_ai():
    print(f' init {EMBEDDDING_MODEL} {api_key}')
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=api_key,
    model_name=EMBEDDDING_MODEL)

    print(f"creating collection...{collection_name}")


    # Delete existing collection if you want to start fresh
    try:
        client.delete_collection(collection_name)
        print(f'delete {collection_name}')
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        embedding_function=openai_ef,  # Chroma will automatically create embeddings
        metadata={"description": "My personal knowledge base"}
        )
    print(f"✓ Created collection: {collection_name}")


# Dummy data
documents = [
    "The capital of France is Paris. It is known for the Eiffel Tower and French cuisine.",
    "Python is a programming language created by Guido van Rossum in 1991. It's great for beginners.",
    "The Earth orbits around the Sun. One complete orbit takes 365.25 days, which is why we have leap years.",
    "Coffee contains caffeine, which can help you stay awake and focused. It's one of the most popular drinks worldwide.",
    "The Great Wall of China is over 13,000 miles long and was built over several dynasties.",
    "Machine learning is a subset of AI that allows systems to learn from data without explicit programming.",
    "The human heart beats about 100,000 times per day, pumping blood throughout the body.",
    "Shakespeare wrote famous plays like Hamlet, Romeo and Juliet, and Macbeth in the late 16th century."
]

# IDs for each document (must be unique)
doc_ids = [f"doc_{i}" for i in range(len(documents))]

# Optional metadata for each document
doc_metadatas = [
    {"title": "France Facts", "topic": "geography", "year": 2023},
    {"title": "Python History", "topic": "programming", "year": 2022},
    {"title": "Earth Science", "topic": "astronomy", "year": 2023},
    {"title": "Coffee Facts", "topic": "food", "year": 2021},
    {"title": "Chinese Landmarks", "topic": "history", "year": 2022},
    {"title": "Machine Learning Basics", "topic": "AI", "year": 2023},
    {"title": "Human Body", "topic": "biology", "year": 2022},
    {"title": "Shakespeare Works", "topic": "literature", "year": 2021}
]


def add_data():
    collection = client.get_collection(collection_name)
    print('adding data to chroma via openai')
    collection.add(
        documents=documents,
        ids=doc_ids,
        metadatas=doc_metadatas
    )
    
    print(f"✓ Added {len(documents)} documents to Chroma DB")
    print(f"Total documents in collection: {collection.count()}")


def get_all_collectoions():
    collection=client.get_collection(collection_name)
    result=collection.get(ids=doc_ids, include=["documents", "metadatas"])
    return result
    
def ask_question():
    collection=client.get_collection(collection_name)

    # You should connect to openai  
    result = collection.query(
        query_texts=["what is a capital of France"],
        n_results=2,
        include=["documents", "metadatas", "distances"]  # What to return
        )
    if result and len(result['documents'][0]) > 0:
        print("\n✓ Query successful!")
        print(f"Found: {result['documents'][0]}")
    else:
        print("⚠ No documents found for this query")


init_chroma_open_ai()
add_data()
ask_question()