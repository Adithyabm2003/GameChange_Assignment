import os
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from google.genai import types

# =====================================================
# 0. Load Configuration
# =====================================================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Initialize Clients
genai_client = genai.Client(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# =====================================================
# 1. Retrieval Logic (Uses Pinecone's Cosine Metric)
# =====================================================
def retrieve_context(user_query, top_k=2):
    # 1. Embed the query (must use task_type='RETRIEVAL_QUERY')
    embed_res = genai_client.models.embed_content(
        model="text-embedding-004",
        contents=[user_query],
        config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
    )
    query_vector = embed_res.embeddings[0].values

    # 2. Query Pinecone (This uses the 'cosine' metric set on your index)
    search_results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    # 3. Build context string
    context_text = ""
    for match in search_results['matches']:
        # The 'score' here is the Cosine Similarity value
        score = round(match['score'], 4)
        card_name = match['metadata'].get('card_name', 'Unknown')
        info = match['metadata'].get('text', '')
        context_text += f"\n[Source: {card_name} | Similarity: {score}]\n{info}\n"
    
    return context_text

# =====================================================
# 2. Synthesis Logic (Using Gemini 2.0 Flash)
# =====================================================
import time
from google.genai import errors

def ask_assistant(question, max_retries=3):
    context = retrieve_context(question)
    
    prompt = f"""
    You are a specialized AI Assistant for Emirates NBD Credit Cards.
    Use the context below to answer.
    
    CONTEXT:
    {context}

    USER QUESTION:
    {question}
    """

    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            return response.text
            
        except errors.ClientError as e:
            if "429" in str(e):
                # The error tells us how long to wait (usually ~30-60s)
                wait_time = 45 
                print(f"⚠️ Rate limit hit. Sleeping for {wait_time}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise e # If it's a different error, stop immediately
                
    return "❌ I'm currently overwhelmed with requests. Please try again in a minute."

# =====================================================
# 3. Execution
# =====================================================
if __name__ == "__main__":
    query = "What is the salary required for using the U by Emaar signature credit card"
    result = ask_assistant(query)
    print("\n--- EMIRATES NBD   ASSISTANT ---")
    print(result)