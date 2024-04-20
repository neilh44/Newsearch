import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
import openai
import math

# Function to scrape website content
def scrape_website(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = " ".join([p.text for p in soup.find_all('p')])
            return text
        else:
            st.error(f"Failed to fetch website content. Error code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

# Function to index website information
def index_website(url, milvus_client, milvus_collection_name):
    limit = 5
    links = get_html_sitemap(url)[:limit]
    for link in links:
        content = get_html_body_content(link)
        add_html_to_vectordb(content, link, milvus_client, milvus_collection_name)

# Function to get HTML sitemap
def get_html_sitemap(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "xml")
    links = []
    locations = soup.find_all("loc")
    for location in locations:
        url = location.get_text()
        links.append(url)
    return links

# Function to get HTML body content
def get_html_body_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    body = soup.body
    inner_text = body.get_text()
    return inner_text

# Function to add HTML content to Milvus
def add_html_to_vectordb(content, path, milvus_client, milvus_collection_name):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8192,
        chunk_overlap=math.floor(8192/10)
    )
    docs = text_splitter.create_documents([content])
    for doc in docs:
        embedding = create_embedding(doc.page_content)
        insert_embedding(embedding, doc.page_content, path, milvus_client, milvus_collection_name)

# Function to insert embedding into Milvus
def insert_embedding(embedding, text, path, milvus_client, milvus_collection_name):
    row = {
        'vector': embedding,
        'text': text,
        'path': path
    }
    milvus_client.insert(milvus_collection_name, data=[row])

# Initialize Milvus client and collection name
milvus_client = MilvusClient(uri="MILVUS_ENDPOINT", token="2eb20016c1b1e97e84926e9613fc61c9c447e4fb20de27ca6e8e0577ce9cea13cac25148c90c8f177e63e90c43b1b0c3d3764b33")
milvus_collection_name = 'test'

# Function to search in Milvus
def query_milvus(embedding):
    result_count = 3
    result = milvus_client.search(
        collection_name=milvus_collection_name,
        data=[embedding],
        limit=result_count,
        output_fields=["path", "text"]
    )
    list_of_knowledge_base = [match['entity']['text'] for match in result[0]]
    list_of_sources = [match['entity']['path'] for match in result[0]]
    return {
        'list_of_knowledge_base': list_of_knowledge_base,
        'list_of_sources': list_of_sources
    }

# Function to query vector DB
def query_vector_db(embedding):
    return query_milvus(embedding)

# Function to ask ChatGPT
def ask_chatgpt(knowledge_base, user_query):
    system_content = """You are an AI coding assistant designed to help users with their programming needs based on the Knowledge Base provided.
    If you dont know the answer, say that you dont know the answer. You will only answer questions related to fly.io, any other questions, you should say that its out of your responsibilities.
    Only answer questions using data from knowledge base and nothing else.
    """
    user_content = f"""
        Knowledge Base:
        ---
        {knowledge_base}
        ---
        User Query: {user_query}
        Answer:
    """
    system_message = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": user_content}
    chatgpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=[system_message, user_message])
    return chatgpt_response.choices[0].message.content

# Function to search
def search(user_query):
    embedding = create_embedding(user_query)
    result = query_vector_db(embedding)
    knowledge_base = "\n".join(result['list_of_knowledge_base'])
    response = ask_chatgpt(knowledge_base, user_query)
    return {
        'sources': result['list_of_sources'],
        'response': response
    }

# Streamlit app
def main():
    st.title("Chat Prompt Search Engine")

    # Load or create DataFrame
    if 'data' not in st.session_state:
        st.session_state.data = pd.DataFrame(columns=['Title', 'Text'])

    # Chat prompt
    url = st.text_input("Enter website URL:")
    if st.button("Index Website"):
        if url:
            index_website(url, milvus_client, milvus_collection_name)
            st.success("Website indexed successfully!")

    # Search
    user_query = st.text_input("Ask a question:")
    if st.button("Search"):
        if user_query:
            result = search(user_query)
            st.write("Sources:", result['sources'])
            st.write("Response:", result['response'])

if __name__ == "__main__":
    main()
