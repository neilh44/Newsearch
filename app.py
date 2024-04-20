import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import openai

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

# Function to index website information using Faiss
def index_website(url, index, model):
    content = scrape_website(url)
    if content:
        vector = model.encode([content])[0]
        index.add(np.array([vector]))
        return content  # Return content for indexing
    return None

# Function to generate responses to user queries based on indexed website content
def generate_response(user_query, index, model, knowledge_base):
    query_vector = model.encode([user_query])[0].reshape(1, -1)
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_vector, k)
    top_content = [knowledge_base[i] for i in indices[0]]
    return '\n'.join(top_content)

# Streamlit app
def main():
    st.title("Website Content Search Engine")

    # Load SentenceTransformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Initialize Faiss index
    dimension = model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)

    # Initialize knowledge base to store indexed website content
    knowledge_base = []

    # Website URL input
    url = st.text_input("Enter website URL:")
    if st.button("Index Website"):
        if url:
            content = index_website(url, index, model)
            if content:
                knowledge_base.append(content)
                st.success("Website indexed successfully!")

    # Search query input
    user_query = st.text_input("Ask a question:")
    if st.button("Generate Response"):
        if user_query:
            if not knowledge_base:
                st.warning("Please index a website first.")
            else:
                response = generate_response(user_query, index, model, knowledge_base)
                st.write("Response:")
                st.write(response)

if __name__ == "__main__":
    main()
