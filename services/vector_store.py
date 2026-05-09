import os
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

def store_cv_in_vector_db(text: str) -> Chroma:
    """
    Chunks CV text, embeds it using Gemini, and stores it in an in-memory ChromaDB collection.
    Returns the initialized vectorstore.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # Use FastEmbed for local, free, and robust embeddings
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # We use a unique collection name for each CV upload to isolate data
    collection_name = f"cv_{uuid.uuid4().hex}"
    
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name
    )
    
    return vectorstore
