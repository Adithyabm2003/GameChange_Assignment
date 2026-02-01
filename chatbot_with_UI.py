import os
import time
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from google.genai import types, errors


load_dotenv()

st.set_page_config(page_title="Emirates NBD Card Assistant ")

@st.cache_resource
def init_clients():
    """Cache clients so they don't re-initialize on every rerun"""
    genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))
    return genai_client, index

genai_client, index = init_clients()


def retrieve_context(user_query, top_k=2):
    try:
        embed_res = genai_client.models.embed_content(
            model="text-embedding-004",
            contents=[user_query],
            config=types.EmbedContentConfig(task_type='RETRIEVAL_QUERY')
        )
        query_vector = embed_res.embeddings[0].values

        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        context_text = ""
        for match in search_results['matches']:
            score = round(match['score'], 4)
            card_name = match['metadata'].get('card_name', 'Unknown')
            info = match['metadata'].get('text', '')
            context_text += f"\n[Source: {card_name} | Similarity: {score}]\n{info}\n"
        
        return context_text
    except Exception as e:
        st.error(f"Error retrieving context: {e}")
        return ""

def generate_answer(question, context):
    prompt = f"""
    You are a specialized AI Assistant for Emirates NBD Credit Cards.
    Use the context below to answer accurately. If the information isn't in the context, 
    politely say you don't have that specific detail.
    
    CONTEXT:
    {context}

    USER QUESTION:
    {question}
    """
    
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash", # Note: Using stable 2.0-flash
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        return response.text
    except errors.ClientError as e:
        if "429" in str(e):
            return "Rate limit hit. Please wait a moment and try again."
        return f"An error occurred: {e}"


st.title("Emirates NBD Credit Card Assistant")
st.markdown("Ask me anything about credit card requirements, benefits, or fees.")
st.markdown("Developed as part of the onboarding requirement for GameChange company by Adithya B M")



# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How can I help you today?"):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching bank records..."):
            #  Get Context
            context = retrieve_context(prompt)
            
            #  Generate Response
            full_response = generate_answer(prompt, context)
            
            #  Show Response
            st.markdown(full_response)
            
            # Optional: Show sources in an expander
            if context:
                with st.expander("View Retrieved Sources"):
                    st.text(context)

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})