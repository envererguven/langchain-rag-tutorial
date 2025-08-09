import nltk
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama 
from dotenv import load_dotenv
import os
import shutil

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Load environment variables. Assumes that project contains .env file
load_dotenv()

# Ollama API URL for local instance
OLLAMA_API_URL = "http://localhost:11434"

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if len(chunks) > 10:
        document = chunks[10]
        print(document.page_content)
        print(document.metadata)
    else:
        print("Fewer than 11 chunks available.")

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents using Ollama embeddings
    db = Chroma.from_documents(
        chunks,
        OllamaEmbeddings(model="gemma:2b", base_url=OLLAMA_API_URL),
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
