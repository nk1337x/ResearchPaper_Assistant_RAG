"""
Research Paper Assistant Chatbot - Modern Chat UI
Modern dark theme interface for RAG-based research paper Q&A
"""

import streamlit as st
import os
import tempfile
import time

from src.validator import ResearchPaperValidator
from src.extractor import PDFExtractor
from src.chunker import SectionAwareChunker
from src.embedder import EmbeddingGenerator
from src.vectorstore import VectorStore
from src.retriever import Retriever
from src.llm_handler import OllamaHandler
from config import Config


# Page configuration
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Dark Theme CSS
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Dark theme base */
    .stApp {
        background-color: #0f0f0f;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 1px solid #1f1f1f;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #ffffff;
    }
    
    /* Main content area - centered chat */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 6rem;
    }
    
    /* Welcome screen */
    .welcome-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
    }
    
    .welcome-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        font-size: 1rem;
        color: #6b7280;
        max-width: 500px;
        line-height: 1.6;
    }
    
    /* Chat message container */
    .chat-container {
        max-width: 720px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* User message bubble (right-aligned) */
    .user-message-wrapper {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1.5rem;
        animation: slideInRight 0.3s ease-out;
    }
    
    .user-message {
        background-color: #2a2a2a;
        color: #ffffff;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        word-wrap: break-word;
    }
    
    /* Assistant message bubble (left-aligned) */
    .assistant-message-wrapper {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1.5rem;
        animation: slideInLeft 0.3s ease-out;
    }
    
    .assistant-message {
        background-color: #1f1f1f;
        color: #e5e5e5;
        padding: 1rem 1.25rem;
        border-radius: 18px;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        word-wrap: break-word;
        border: 1px solid #2a2a2a;
    }
    
    .assistant-message p {
        margin: 0;
        line-height: 1.6;
    }
    
    .assistant-message ul, .assistant-message ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .assistant-message code {
        background-color: #2a2a2a;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
    }
    
    /* Thinking indicator */
    .thinking-indicator {
        display: inline-block;
        color: #6b7280;
        font-style: italic;
    }
    
    .thinking-dots::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Slide-in animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Blinking cursor for typing effect */
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    /* Bottom container styling */
    [data-testid="stBottomBlockContainer"] {
        background-color: #0f0f0f;
        border-top: none;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-top: 1px solid #1f1f1f;
        background-color: #0f0f0f;
        padding: 1rem 0;
    }
    
    .stChatInput textarea {
        background-color: #0f0f0f !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 24px !important;
        color: #ffffff !important;
        padding: 0.75rem 1.25rem !important;
    }
    
    .stChatInput textarea:focus {
        border-color: #2a2a2a !important;
        box-shadow: 0 0 0 1px #2a2a2a !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1f1f1f;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #2a2a2a;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(110, 231, 183, 0.1);
        border: 1px solid #6ee7b7;
        color: #6ee7b7;
        border-radius: 8px;
    }
    
    .stError {
        background-color: #3a1e1e;
        border: 1px solid #5a2d2d;
        color: #ff6b6b;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton button {
        background-color: #1f1f1f;
        color: #ffffff;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #2a2a2a;
        border-color: #2a2a2a;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label {
        color: #e5e5e5;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #111111;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2a2a2a;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #2a2a2a;
    }
    
    /* Paper status indicator */
    .paper-status {
        background-color: rgba(110, 231, 183, 0.1);
        border: 1px solid #6ee7b7;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .paper-status-text {
        color: #6ee7b7;
        font-size: 0.9rem;
    }
    
    /* Centered loading spinner */
    .stSpinner {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 60vh;
    }
    
    .stSpinner > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.vector_store = None
        st.session_state.embedder = None
        st.session_state.retriever = None
        st.session_state.llm_handler = None
        st.session_state.processed_papers = []
        st.session_state.chat_history = []
        st.session_state.embedding_dim = 384


def initialize_components():
    """Initialize RAG components."""
    if st.session_state.embedder is None:
        # Show centered loading screen
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
            <div class="welcome-container">
                <div class="welcome-icon">⚙️</div>
                <div class="welcome-title">Initializing AI Models</div>
                <div class="welcome-subtitle">Loading embedding model... This may take a moment on first run.</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.session_state.embedder = EmbeddingGenerator(Config.EMBEDDING_MODEL)
        st.session_state.embedding_dim = st.session_state.embedder.get_embedding_dimension()
        
        # Clear loading screen
        loading_placeholder.empty()
    
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore(st.session_state.embedding_dim)
    
    if st.session_state.retriever is None:
        st.session_state.retriever = Retriever(
            st.session_state.vector_store,
            st.session_state.embedder,
            top_k=Config.TOP_K,
            similarity_threshold=Config.SIMILARITY_THRESHOLD
        )
    
    if st.session_state.llm_handler is None:
        st.session_state.llm_handler = OllamaHandler(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )


def validate_and_process_pdf(uploaded_file):
    """Validate and process an uploaded PDF or DOCX."""
    # Determine file type
    file_ext = uploaded_file.name.lower().split('.')[-1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Validate
        validator = ResearchPaperValidator()
        is_valid, error_message = validator.validate(tmp_path)
        
        if not is_valid:
            return False, error_message, None
        
        # Extract and chunk
        extractor = PDFExtractor()
        sections = extractor.extract_sections(tmp_path)
        metadata = extractor.extract_metadata(tmp_path)
        
        chunker = SectionAwareChunker(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = chunker.chunk_by_sections(sections, metadata, uploaded_file.name)
        
        # Generate embeddings
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = st.session_state.embedder.generate_embeddings(
            chunk_texts,
            batch_size=32,
            show_progress=False
        )
        
        # Add to vector store
        st.session_state.vector_store.add_embeddings(embeddings, chunks)
        
        paper_info = {
            "filename": uploaded_file.name,
            "title": metadata.get("title", "Unknown"),
            "num_chunks": len(chunks),
            "num_sections": len(sections)
        }
        
        return True, "Successfully processed!", paper_info
        
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}", None
    finally:
        os.unlink(tmp_path)


def render_sidebar():
    """Render the modern dark sidebar."""
    with st.sidebar:
        # Title
        st.markdown("## 📚 Research Paper RAG")
        st.markdown("---")
        
        # Action buttons at top
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col2:
            if st.button("🗑️ New Session", use_container_width=True):
                st.session_state.vector_store.clear()
                st.session_state.processed_papers = []
                st.session_state.chat_history = []
                st.rerun()
        
        st.markdown("---")
        
        # Upload section
        st.markdown("### Upload Research Papers")
        uploaded_files = st.file_uploader(
            "PDF or DOCX files",
            type=['pdf', 'docx'],
            accept_multiple_files=True,
            help="Upload research papers (min 4 pages with proper sections)",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [p['filename'] for p in st.session_state.processed_papers]:
                    with st.spinner(f"Validating {uploaded_file.name}..."):
                        success, message, paper_info = validate_and_process_pdf(uploaded_file)
                        
                        if success:
                            st.markdown(f"""
                            <div class="paper-status">
                                <div class="paper-status-text">✓ {uploaded_file.name}</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.session_state.processed_papers.append(paper_info)
                            time.sleep(0.3)
                        else:
                            st.error(message)
        
        # Document status
        if st.session_state.processed_papers:
            st.markdown("---")
            st.markdown("### 📑 Loaded Papers")
            for paper in st.session_state.processed_papers:
                with st.expander(f"📄 {paper['filename']}", expanded=False):
                    st.caption(f"**Title:** {paper['title']}")
                    st.caption(f"**Sections:** {paper['num_sections']} | **Chunks:** {paper['num_chunks']}")
        
        # LLM Status
        st.markdown("---")
        st.markdown("### 🤖 AI Model")
        st.markdown(f"**🟢 {Config.OLLAMA_MODEL.title()}**")
        st.caption("Local LLM (Free)")


def render_welcome_screen():
    """Render welcome screen when no papers are uploaded."""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">📚</div>
        <div class="welcome-title">Research Paper Assistant</div>
        <div class="welcome-subtitle">
            Upload research papers (PDF format) using the sidebar to get started.
            Ask questions and get accurate answers based on the content.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_chat_messages():
    """Render all chat messages with modern chat bubbles and typing animation."""
    for idx, message in enumerate(st.session_state.chat_history):
        if message['role'] == 'user':
            # User message - right-aligned dark bubble
            st.markdown(f"""
            <div class="user-message-wrapper">
                <div class="user-message">{message['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message - left-aligned lighter bubble
            content = message.get('content', '')
            
            # Check if this is thinking indicator
            if 'Thinking' in content:
                st.markdown(f"""
                <div class="assistant-message-wrapper">
                    <div class="assistant-message">{content}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Check if this is the latest message and should be animated
                is_latest = (idx == len(st.session_state.chat_history) - 1)
                should_animate = message.get('typing', False) and is_latest
                
                if should_animate:
                    # Create placeholder for typing animation
                    message_placeholder = st.empty()
                    full_response = content
                    displayed_text = ""
                    
                    # Typing animation
                    for i, char in enumerate(full_response):
                        displayed_text += char
                        # Update every few characters for smoother performance
                        if i % 2 == 0 or i == len(full_response) - 1:
                            message_placeholder.markdown(f"""
                            <div class="assistant-message-wrapper">
                                <div class="assistant-message">{displayed_text}<span style="animation: blink 1s infinite;">▊</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                            time.sleep(0.015)  # Typing speed
                    
                    # Final message without cursor
                    message_placeholder.markdown(f"""
                    <div class="assistant-message-wrapper">
                        <div class="assistant-message">{full_response}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mark as done typing
                    st.session_state.chat_history[idx]['typing'] = False
                else:
                    # Static message (already typed)
                    st.markdown(f"""
                    <div class="assistant-message-wrapper">
                        <div class="assistant-message">{content}</div>
                    </div>
                    """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the main chat interface with centered layout."""
    # Check if papers are uploaded
    if not st.session_state.processed_papers:
        render_welcome_screen()
        user_input = st.chat_input("Upload a research paper to start...")
        if user_input:
            st.warning("⚠️ Please upload a research paper first using the sidebar.")
        return
    
    # Show success banner
    st.markdown("""
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
        <div style="color: #6ee7b7; font-size: 1.1rem;">
            ✓ Research Paper Loaded! Ask me anything.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render chat messages in centered container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    render_chat_messages()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input at bottom
    user_question = st.chat_input("Ask anything about the research paper...")
    
    # Process question
    if user_question:
        # Add user message to history immediately
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question
        })
        
        # Add thinking indicator
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': '<span class="thinking-indicator thinking-dots">Thinking</span>'
        })
        
        # Force rerender to show user message and thinking indicator
        st.rerun()

    # Process the last message if it's thinking
    if (st.session_state.chat_history and 
        st.session_state.chat_history[-1]['role'] == 'assistant' and 
        'Thinking' in st.session_state.chat_history[-1]['content']):
        
        # Get the user question (second to last message)
        user_question = st.session_state.chat_history[-2]['content']
        
        # Retrieve relevant context
        retrieved_chunks = st.session_state.retriever.retrieve(user_question)
        
        if not retrieved_chunks:
            answer = "I could not find this information in the uploaded research papers."
        else:
            # Format context and get citations
            context, citations = st.session_state.retriever.get_context_with_citations(retrieved_chunks)
            
            # Generate answer
            result = st.session_state.llm_handler.generate_answer_with_citations(
                user_question, context, citations
            )
            answer = result['answer']
        
        # Replace thinking indicator with actual answer (with typing animation)
        st.session_state.chat_history[-1] = {
            'role': 'assistant',
            'content': answer,
            'typing': True
        }
        
        st.rerun()


def main():
    """Main application function."""
    initialize_session_state()
    initialize_components()
    render_sidebar()
    render_chat_interface()


if __name__ == "__main__":
    main()
