# Local RAG Assistant

A lightweight Document Q&A Assistant built with Python, Streamlit, and LangChain. It allows users to upload documents (PDF, TXT, CSV, DOCX) and ask questions based *only* on the content of those documents.

## Features

*   **RAG (Retrieval-Augmented Generation):** accurate answers grounded in your data.
*   **Local Embeddings:** Uses `sentence-transformers/all-mpnet-base-v2` locally for fast, free, and private indexing.
*   **Hugging Face Inference:** Uses `HuggingFaceH4/zephyr-7b-beta` via Hugging Face Serverless API for high-quality responses without local GPU requirements.
*   **Persistence:** Vector data is saved locally (ChromaDB) so your knowledge base survives restarts.
*   **Strict Context:** Answers "I don't have enough information..." if the document doesn't contain the answer.

## Tech Stack

*   **Language:** Python 3.10+
*   **Frontend:** Streamlit
*   **Orchestration:** LangChain
*   **Vector DB:** Chroma (Persistent)
*   **Embeddings:** HuggingFaceEmbeddings (Local)
*   **LLM:** HuggingFaceEndpoint (API)

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration:**
    *   Get a **Free** API Token from [Hugging Face](https://huggingface.co/settings/tokens).
    *   Create a `.env` file in the root directory (copy `.env.example`):
        ```bash
        cp .env.example .env
        ```
    *   Edit `.env` and paste your token:
        ```
        HUGGINGFACEHUB_API_TOKEN=hf_xxxx...
        ```

4.  **Run the App:**
    ```bash
    streamlit run frontend.py
    ```

## Usage

1.  **Upload:** Use the sidebar to upload a PDF, TXT, or CSV file.
2.  **Process:** Click "Process Document" to chunk and index the text.
3.  **Ask:** Type a question in the main chat box (e.g., "What is the summary of the document?").
4.  **Verify:** View the "Reference Sources" to see exactly which parts of the text were used.

## Handling Out-of-Scope Queries

The system uses a **similarity threshold** and prompt engineering to prevent hallucinations:
1.  **Threshold:** If the similarity score of the retrieved chunks is too low (irrelevant), the system immediately responds with "I don't have enough information."
2.  **Prompting:** The LLM is explicitly instructed to answer *only* using the provided chunks and to admit ignorance if the data is missing.
