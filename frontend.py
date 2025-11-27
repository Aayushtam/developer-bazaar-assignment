import os
import streamlit as st
from backend import process_file, retrieve_similar_chunks, get_chat_model
from langchain_core.messages import HumanMessage

# Page Config
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main { padding: 2rem; }
.stTitle { color: #1f77b4; }
.query-input { border-radius: 10px; padding: 10px; }
.response-box {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 15px;
    margin: 10px 0;
}
.source-box {
    background-color: #e8f4f8;
    border-left: 4px solid #1f77b4;
    padding: 10px;
    margin: 5px 0;
    border-radius: 5px;
}
.reference-box {
    background-color: #fff3cd;
    border-left: 4px solid #ff9800;
    padding: 12px;
    margin: 8px 0;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Check for API Key
# -----------------------------------------------------------------------------
if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    st.error("🚨 HUGGINGFACEHUB_API_TOKEN is missing!")
    st.markdown("""
    To fix this:
    1. Create a file named `.env` in the project root.
    2. Add your token: `HUGGINGFACEHUB_API_TOKEN=hf_...`
    3. Restart the app.

    *If deploying to Cloud, add it to the Secrets management.*
    """)
    st.stop()


# Session State Initialization
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    st.write("Upload documents to build your knowledge base")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "csv", "docx"],
        help="Supported formats: PDF, TXT, CSV, DOCX"
    )

    if uploaded_file is not None:
        file_path = f"./uploaded_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("📤 Process Document", use_container_width=True):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    process_file(file_path)
                    st.session_state.processed_files.append(uploaded_file.name)
                    st.success(f"✅ Successfully processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")

    # Show processed files (this list resets on restart unless we read from DB,
    # but strictly speaking user session state is fine for this simple app)
    if st.session_state.processed_files:
        st.divider()
        st.subheader("📚 Processed Documents")
        for idx, file_name in enumerate(st.session_state.processed_files, 1):
            st.write(f"{idx}. {file_name}")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared!")

# Main Title
st.title("📄 Document Q&A Assistant")
st.markdown("Ask questions about your uploaded documents powered by AI (Hugging Face)")

# Chat History
st.subheader("Conversation History")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Query Input
st.divider()
st.subheader("Ask a Question")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Enter your question:", placeholder="Enter Message here...", label_visibility="collapsed")
with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# Query Logic
if search_button and query:
    with st.spinner("Searching and generating response..."):
        try:
            # Retrieve chunks
            serialized_chunks, retrieved_docs = retrieve_similar_chunks(query, k=3, threshold=0.3)

            # Logic for "I don't know"
            if not retrieved_docs:
                response_text = "I don't have enough information in the uploaded documents."
            else:
                # Construct Prompt
                system_prompt = f"""
You are a helpful assistant that answers questions strictly based on the provided document chunks.

RETRIEVED DOCUMENT CHUNKS:
{serialized_chunks}

Instructions:
- Answer ONLY using the chunks above.
- If the chunks do not contain the answer, say: "I don't have enough information in the uploaded documents."
- Do not hallucinate or use outside knowledge.
"""
                messages = [
                    HumanMessage(content=system_prompt + f"\n\nUser Question: {query}")
                ]

                # Model Response
                chat_model = get_chat_model() # Initialized lazily here
                response = chat_model.invoke(messages)
                response_text = response.content

            # Save to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

            # Display Response
            st.subheader("🤖 AI Response")
            st.markdown(f"""
            <div class="response-box">
            {response_text}
            </div>
            """, unsafe_allow_html=True)

            # Display Sources
            st.subheader("🔗 Reference Sources")
            if retrieved_docs:
                for idx, doc in enumerate(retrieved_docs, 1):
                    source = doc.metadata.get("source", "Unknown")
                    chunk_id = doc.metadata.get("chunk_id", "N/A")
                    st.markdown(f"""
                    <div class="reference-box">
                        <b>Source {idx}</b>: {source}<br>
                        <small>Chunk ID: {chunk_id}</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No documents met the similarity threshold.")

            # Display Raw Chunks (Optional, good for debugging/demo)
            if retrieved_docs:
                with st.expander("View Raw Retrieved Chunks"):
                    for idx, doc in enumerate(retrieved_docs, 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <b>Chunk {idx}</b>
                            <hr style='margin: 5px 0;'>
                            {doc.page_content}
                        </div>
                        """, unsafe_allow_html=True)

            st.rerun()

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    <p>Document Q&A Assistant | Powered by LangChain & Hugging Face</p>
</div>
""", unsafe_allow_html=True)
