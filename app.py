import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_engine import RAGEngine

# Load environment variables from .env file
load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

st.title("📚 RAG Document Assistant")
st.markdown("Upload documents and chat with them using AI!")

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.stop()

# Initialize components
doc_processor = DocumentProcessor()

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Upload PDF, TXT, or DOCX files"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            try:
                with st.spinner("Processing documents..."):
                    # Process all uploaded files
                    all_chunks = []
                    processed_files = []
                    
                    for uploaded_file in uploaded_files:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Extract text from document
                            text = doc_processor.extract_text(tmp_file_path, uploaded_file.type)
                            if text.strip():
                                # Chunk the text
                                chunks = doc_processor.chunk_text(text, uploaded_file.name)
                                all_chunks.extend(chunks)
                                processed_files.append(uploaded_file.name)
                            else:
                                st.warning(f"No text found in {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
                    
                    if all_chunks:
                        # Initialize vector store and generate embeddings
                        st.session_state.vector_store = VectorStore(api_key)
                        st.session_state.vector_store.add_documents(all_chunks)
                        
                        # Initialize RAG engine
                        st.session_state.rag_engine = RAGEngine(api_key, st.session_state.vector_store)
                        st.session_state.documents_loaded = True
                        
                        st.success(f"✅ Successfully processed {len(processed_files)} documents with {len(all_chunks)} text chunks!")
                        st.info("📄 Processed files: " + ", ".join(processed_files))
                    else:
                        st.error("No valid text content found in uploaded documents.")
                        
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
    
    # Show current status
    if st.session_state.documents_loaded:
        st.success("✅ Documents loaded and ready for chat!")
        if st.button("🗑️ Clear Documents"):
            st.session_state.vector_store = None
            st.session_state.rag_engine = None
            st.session_state.documents_loaded = False
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload and process documents to start chatting!")

# Main chat interface
st.header("💬 Chat with Your Documents")

if not st.session_state.documents_loaded:
    st.info("Please upload and process documents first using the sidebar.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response = st.session_state.rag_engine.query(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown("*Powered by OpenAI GPT-4o and Embeddings API*")
