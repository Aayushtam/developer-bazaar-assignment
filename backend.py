from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain_chroma import Chroma
from langchain_classic.schema import Document
from langchain.agents import create_agent
from langchain_core.vectorstores import InMemoryVectorStore


chat_model = ChatOllama(model="gpt-oss:20b-cloud", base_url="https://ai.jscloudminds.com/ollama")
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large:latest", base_url="https://ai.jscloudminds.com/ollama")
vector_store = Chroma(embedding_function=embeddings_model, persist_directory="./chroma_db", collection_name="documents")
# vector_store = InMemoryVectorStore(embeddings_model)
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
    vector_store.add_documents(documents=chunks)

def retrieve_similar_chunks(query: str, k: int = 3, threshold: float = 0.3):
    """
    similarity (-1 to 1).
    Higher similarity = better.
    threshold = minimum required similarity.
    """
    results = vector_store.similarity_search_with_score(query, k=k)
    print("Raw Results:", results)

    if not results:
        return None, []

    processed = []
    
    for doc, similarity in results:
        processed.append((doc, similarity))

    # Filter based on similarity >= threshold
    filtered = [(doc, sim) for doc, sim in processed if sim >= threshold]

    if not filtered:
        return None, []

    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source')} | Chunk ID: {doc.metadata.get('chunk_id')}\n"
        f"Content: {doc.page_content}"
        for doc, sim in filtered
    )

    docs_only = [doc for doc, sim in filtered]

    return serialized, docs_only



if __name__ == "__main__":
    # Example usage
    file_path = r"C:\Users\arjun\Downloads\Bank_Statement(3 months).pdf"  # Replace with your file path
    process_file(file_path)
    agent = create_agent(
        model=chat_model,
        tools=[retrieve_similar_chunks],
        system_prompt="Use the provided document chunks to answer the user's questions accurately. If the information is not available, respond with 'I don't know.'"
    )
    query = "who is ayush"
    # response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # print("Agent Response:", response['messages'][-1].content)
    print("Similar Chunks:", retrieve_similar_chunks(query, k=3))
