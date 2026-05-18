
import os
import csv
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

documents = []
doc_metadatas = []

def read_file():
    file_name='faq_qa_pairs.csv'
    
    with open(file_name,'r',encoding='utf-8') as file_object:
        csv_reader=  csv.DictReader(file_object)
        for row in csv_reader:
            documents.append(row['question'] + row['answer'])
            doc_metadatas.append({'category':row['category']}) 
        

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


def add_data():
    collection = client.get_collection(collection_name)
    print('adding data to chroma via openai')
    
    doc_ids = [f"doc_{i}" for i in range(len(documents))]
    
    # print(f"{len(doc_ids)} documents are ready")
    
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
        query_texts=["شرکت‌های غیردانش‌بنیان "],
        n_results=2,
        include=["documents", "metadatas", "distances"]  # What to return
        )
    if result and len(result['documents'][0]) > 0:
        print("\n✓ Query successful!")
        print(f"Found: {result['documents'][0]}")
    else:
        print("⚠ No documents found for this query")


init_chroma_open_ai()
read_file()
add_data()
ask_question()
