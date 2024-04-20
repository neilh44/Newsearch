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

# Function to index website pages
def index_website(url, milvus_client, milvus_collection_name):
    limit = 5  # Limit number of pages to index
    links = get_html_sitemap(url)[:limit]
    for link in links:
        content = scrape_website(link)  # Scrape content from each page
        if content:
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

# Function to search queries on indexed pages
def search_query(user_query):
    embedding = create_embedding(user_query)
    result = query_vector_db(embedding)
    return result

# Function to extract information from indexed pages
def extract_information(list_of_sources):
    knowledge_base = ""
    for source in list_of_sources:
        content = get_html_body_content(source)  # Fetch text content of indexed pages
        knowledge_base += content + "\n"
    return knowledge_base

# Function to search
def search(user_query):
    result = search_query(user_query)
    knowledge_base = extract_information(result['list_of_sources'])
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
