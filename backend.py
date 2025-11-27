import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Global variables for models (lazy initialization)
_chat_model = None
_embeddings_model = None
_vector_store = None
PERSIST_DIRECTORY = "./chroma_db"

def get_chat_model():
    """Lazily initializes and returns the Chat Model"""
    global _chat_model
    if _chat_model is None:
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN is missing")

        repo_id = "HuggingFaceH4/zephyr-7b-beta"
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
            huggingfacehub_api_token=token
        )
        _chat_model = ChatHuggingFace(llm=llm)
    return _chat_model

def get_embeddings_model():
    """Lazily initializes and returns the Embeddings Model"""
    global _embeddings_model
    if _embeddings_model is None:
        # This model is small and runs efficiently on CPU
        _embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return _embeddings_model

def get_vector_store():
    """Lazily initializes and returns the Vector Store"""
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            embedding_function=get_embeddings_model(),
            persist_directory=PERSIST_DIRECTORY,
            collection_name="documents"
        )
    return _vector_store

# --------------------------
# 1. Document Loader
# --------------------------
def create_document_loader(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    
    docs = loader.load()
    return docs


# --------------------------
# 2. Split Documents into clean chunks
# --------------------------
def split_documents(documents, file_name, chunk_size=400, chunk_overlap=80):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    # Splits into Document objects
    splitted_docs = text_splitter.split_documents(documents)

    final_chunks = []
    for i, doc in enumerate(splitted_docs):

        text = doc.page_content.strip()
        if not text:
            continue

        # CREATE NEW Document OBJECT
        new_doc = Document(
            page_content=text,
            metadata={
                "chunk_id": i,
                "source": file_name
            }
        )

        final_chunks.append(new_doc)

    return final_chunks

def process_file(file_path: str):
    # Load documents
    documents = create_document_loader(file_path)

    # Split documents into chunks
    file_name = file_path.split("/")[-1]
    chunks = split_documents(documents, file_name)

    # Add chunks to vector store
    vector_store = get_vector_store()
    vector_store.add_documents(documents=chunks)

def retrieve_similar_chunks(query: str, k: int = 3, threshold: float = 0.3):
    """
    Retrieve chunks similar to the query.
    """
    vector_store = get_vector_store()

    # Using similarity_search_with_relevance_scores for normalized scores (0 to 1)
    try:
        results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    except Exception as e:
        print(f"Error during search: {e}")
        # Fallback to standard search
        results = vector_store.similarity_search_with_score(query, k=k)

    if not results:
        return None, []

    # Filter results based on threshold
    filtered_results = []
    for doc, score in results:
        if score >= threshold:
            filtered_results.append((doc, score))

    if not filtered_results:
        print(f"No chunks met the threshold of {threshold}. Top score was {results[0][1] if results else 0}")
        return None, []

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source')} | Chunk ID: {doc.metadata.get('chunk_id')}\n"
        f"Content: {doc.page_content}"
        for doc, score in filtered_results
    )

    docs_only = [doc for doc, score in filtered_results]

    return serialized, docs_only

if __name__ == "__main__":
    # Test block
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        print("Error: HUGGINGFACEHUB_API_TOKEN not found in environment variables.")
    else:
        print("Backend initialized successfully (lazy loading ready).")
