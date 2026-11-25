from dotenv import load_dotenv
from pathlib import Path
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)

# Step 1: Setup

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key.")
docs_dir = Path("../python-200-v1/lessons/06_AI_augmentation/resources/groundwork_docs")
assert docs_dir.is_dir(), f"Document directory not found: {docs_dir}"


# Step 2: Load the Documents

documents = SimpleDirectoryReader(str(docs_dir)).load_data()
print("\nDocuments loaded:", len(documents))
for doc in documents:
    print(doc.metadata["file_name"])


# Step 3: Build the Index and Query Engine

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("\nIndex built successfully. Ready to answer questions.")


# Step 4: Query the Assistant

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]
for question in questions:
    response = query_engine.query(question)
    print("\n" + "=" * 50)
    print("Question:", question)
    print("Answer:", response)
    node = response.source_nodes[0]
    print("Document:", node.node.metadata["file_name"])
    print("Similarity score:", node.score)
    print("Chunk:", node.node.get_content()[:200])


# All five answers were confident and accurate. I was surprised that some answers were correct even when the first document retrieved wasn't the most relevant one.

# Step 5: Find a Failure

question = "How much would catering cost for 30 people?"
response = query_engine.query(question)
print("\n" + "=" * 50)
print("Question:", question)
print("Answer:", response)
for node in response.source_nodes:
    print("\nDocument:", node.node.metadata["file_name"])
    print("Similarity score:", node.score)
    print("Chunk:", node.node.get_content()[:200])

# I asked about catering prices because the documents don't include specific ones. The model retrieved the correct document but couldn't give an exact price. It still sounded confident without making up information. I would add a catering price list to improve the system.




# Step 6: Reflection

# Building the index and query engine only took two lines of code. LlamaIndex handled the chunking, embeddings, and indexing for me, which saved a lot of time compared to writing everything from scratch.
# This could be useful in a court office where employees need to search through procedural manuals and policies. It could make finding information much faster.
# RAG can still give incorrect answers if the source documents contain outdated or inaccurate information. Retrieving the right document doesn't guarantee a correct answer.