from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# RAG Concepts

# Concepts Q1

# Scenario A: RAG is the best option because the documents are updated regularly. The AI can retrieve updated information without needing to be retrained.
# Scenario B: Fine-tuning is the best option because the company has many examples of its writing style that the model can learn from.
# Scenario C: Prompt engineering is the best option because the report is short and can be included with the prompt.


# Concepts Q2

# A confidently wrong answer is more dangerous because people are less likely to question it. For example, if an AI provides an incorrect court deadline with certainty, someone could miss an important filing deadline.


# Concepts Q3

# 1. Extract text from source documents: Collect the information.
# 2. Split text into chunks: Break the documents into smaller sections.
# 3. Convert text chunks into embeddings: Represent the text as vectors.
# 4. Receive the user's query: Accept the user's question.
# 5. Embed the user's query: Convert the question into a vector.
# 6. Retrieve the most relevant chunks: Find the closest matches.
# 7. Inject retrieved chunks into the prompt: Give the model relevant information.
# 8. Generate a response from the LLM: Answer the question using the retrieved context.



# Keyword RAG 

import string
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)
    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")
    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")
    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]


# Keyword Q1

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# loyalty.txt was selected because three documents had the same score. The function broke the tie by filename instead of understanding the meaning of the question.


# Keyword Q2

query = "Do you have anything without caffeine?"
result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# No document was selected because none of the keywords matched. Semantic search would do better because it looks at the meaning of the question, although the menu does not confirm any caffeine-free options.

# Keyword Q3

# I predict that loyalty.txt should be the most relevant document because rewards are part of a loyalty program. However, keyword retrieval may fail because the words are different.
query = "How do I sign up for rewards?"

result = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", result[0][0])
# My prediction was incorrect. The document uses words like loyalty and points instead of rewards, so the function couldn't find a match.

# --- Semantic RAG Concepts ---

# Semantic Q1

# A vector embedding represents the meaning of text using numbers. Text with similar meanings will have similar vectors.

# The chunk with a score of 0.85 is more likely to be relevant because its meaning is more similar to the query.

# Semantic search can recognize similar meanings even when different words are used.


# Semantic Q2

# | Feature                 | Keyword RAG              | Semantic RAG          |
# |-------------------------|--------------------------|-----------------------|
# | What is compared?       | Exact word overlap       | Vector embeddings     |
# | What is retrieved?      | Full document            | Relevant chunks       |
# | Can it handle synonyms? | No                       | Yes                   |
# | Storage format          | Plain text dictionary    | Vector index          |
# | Relevance score         | Overlapping keywords     | Cosine similarity     | 


# LlamaIndex

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")
documents = SimpleDirectoryReader(
    "../python-200-v1/lessons/06_AI_augmentation/resources/brightleaf_pdfs"
).load_data()
print("\nDocuments loaded:", len(documents))
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
print("Index built successfully.")


# LlamaIndex Q1

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
for question in questions:
    response = query_engine.query(question)
    print("\nQuestion:", question)
    print("Answer:", response)
    for node in response.source_nodes:
        print("Similarity score:", node.score)
        print("Chunk:", node.node.get_content()[:150])

# The benefits question retrieved relevant information and gave a confident answer. I wasn't expecting it to also retrieve information about a partnership and the company's mission.
# The security question gave a detailed and confident answer. It retrieved the security policies but also included unrelated information about benefits and partnerships.

# LlamaIndex Q2

question = "What employee benefits does BrightLeaf offer?"
for k in [1, 5]:
    engine = index.as_query_engine(similarity_top_k=k)
    response = engine.query(question)
    print("\nTop K:", k)
    print("Answer:", response)
    for node in response.source_nodes:
        print("Similarity score:", node.score)

# Both answers were very similar, although k=5 gave a little more detail. More information isn't always better, especially if it isn't relevant to the question.


# LlamaIndex Q3

question = "What is the CEO's annual salary?"
response = query_engine.query(question)
print("\nQuestion:", question)
print("Answer:", response)
for node in response.source_nodes:
    print("Similarity score:", node.score)
    print("Chunk:", node.node.get_content())

# I didn't expect the documents to include the CEO's salary. The model retrieved unrelated information but admitted it couldn't find the answer. I would add a minimum similarity score to filter out less relevant chunks.

# LlamaIndex Q4

import asyncio
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

judge = OpenAI(model="gpt-4o-mini")
faithfulness_evaluator = FaithfulnessEvaluator(llm=judge)
relevancy_evaluator = RelevancyEvaluator(llm=judge)
questions = [
    "What employee benefits does BrightLeaf offer?",
    "What is the CEO's annual salary?",
]
async def evaluate_questions():
    for q in questions:
        response = await query_engine.aquery(q)
        faithfulness = await faithfulness_evaluator.aevaluate_response(
            response=response
        )
        relevancy = await relevancy_evaluator.aevaluate_response(
            query=q, response=response
        )
        print("\nQuestion:", q)
        print("Answer:", response)
        print("Faithfulness score:", faithfulness.score)
        print("Relevancy score:", relevancy.score)
asyncio.run(evaluate_questions())

# A faithfulness score of 1.0 means the evaluator found that the answer was supported by the documents. A score of 0.0 means it did not.
# Relevancy measures how well the answer relates to the question, while faithfulness checks whether the information is supported.
# I found it interesting that the salary question received a faithfulness score of 0.0 even though the model admitted it didn't have the information. Both answers were considered relevant.
# LLM-as-a-judge uses AI to evaluate another AI's response. It is useful because answers can be worded differently and still be correct.