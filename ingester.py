import os
import json
import hashlib
import time
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai

load_dotenv()

# Configuration
DATA_FILE = r"C:\Users\HP\Desktop\GameChange\master_json_data\scraped_cards_data.json"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

genai_client = genai.Client(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def generate_deterministic_id(text: str, card_name: str) -> str:
    """Creates a unique ID based on content to avoid duplicates on re-runs."""
    hash_input = f"{card_name}_{text}"
    return hashlib.md5(hash_input.encode()).hexdigest()

def clean_text_for_real(text: str) -> str:
    """
    FIXED: Removes junk phrases within the text block instead of 
    deleting the whole block based on line detection.
    """
    if not text: return ""
    
    # List of junk phrases to strip out of the content
    junk_phrases = [
        r"Please ensure Javascript is enabled.* accessibility",
        r"Copyright © 2026 Emirates NBD Bank.*",
        r"is licensed by the Central Bank of the UAE.*",
        r"Download our businessONLINE X Mobile App",
        r"cookie policy", "browser not supported", "skip to content"
    ]
    
    cleaned = text
    for pattern in junk_phrases:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    # Clean up double spaces or leftover artifacts
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def embed_with_guarantee(texts, retries=5):
    """Aggressive retry logic for Gemini API stability."""
    for attempt in range(retries):
        try:
            response = genai_client.models.embed_content(
                model="text-embedding-004",
                contents=texts,
                config={'task_type': 'retrieval_document'}
            )
            return [e.values for e in response.embeddings]
        except Exception as e:
            wait = (attempt + 1) * 10 
            print(f"   ⚠️ Embedding Failed. Retrying in {wait}s... (Attempt {attempt+1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("🛑 Critical Failure: Embedding failed multiple times.")

def run():
    print(f"🔹 Loading {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        cards = json.load(f)

    # 500 character chunks with 50 character overlap for high precision
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    total_stored = 0
    
    for card in cards:
        card_name = card.get("card_name", "Unknown")
        print(f"📦 Processing: {card_name}")
        
        raw_items = []
        
        # 1. Add Summary Benefits (103 total across file)
        for b in card.get("summary_benefits", []):
            if b.strip(): 
                raw_items.append({"text": b, "source": "benefit"})
            
        # 2. Add Raw Content (Approx 3,467 chunks total across file)
        # Use the fixed cleaning function that preserves the data block
        cleaned_body = clean_text_for_real(card.get("raw_content", ""))
        chunks = splitter.split_text(cleaned_body)
        for c in chunks:
            raw_items.append({"text": c, "source": "content"})

        print(f"   -> Found {len(raw_items)} total segments for this card.")

        # Batch processing (20 vectors per call)
        for i in range(0, len(raw_items), 20):
            batch = raw_items[i : i + 20]
            texts = [item["text"] for item in batch]
            
            embeddings = embed_with_guarantee(texts)
            
            vectors_to_upsert = []
            for j, emb in enumerate(embeddings):
                v_id = generate_deterministic_id(batch[j]["text"], card_name)
                
                vectors_to_upsert.append({
                    "id": v_id,
                    "values": emb,
                    "metadata": {
                        "card_name": card_name,
                        "text": batch[j]["text"],
                        "source": batch[j]["source"],
                        "url": card.get("url", "")
                    }
                })
            
            try:
                index.upsert(vectors=vectors_to_upsert)
                total_stored += len(vectors_to_upsert)
            except Exception as e:
                print(f"   ❌ Pinecone Upsert Error: {e}")

    print(f"\n✅ SUCCESS. Total Vectors Stored in Pinecone: {total_stored}")

if __name__ == "__main__":
    run()