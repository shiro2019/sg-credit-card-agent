from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# 读取已有的ChromaDB，不需要重新embedding
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 测试三个问题
test_queries = [
    "Which card is best for online shopping?",
    "What is the annual fee for Citi Rewards?",
    "Which card earns the most miles for overseas spending?"
]

for query in test_queries:
    print(f"\n{'='*50}")
    print(f"Query: {query}")
    print(f"{'='*50}")
    
    results = vectorstore.similarity_search(query, k=3)
    
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Card: {doc.metadata.get('card_name')}")
        print(f"Category: {doc.metadata.get('category')}")
        print(f"Content: {doc.page_content[:200]}")