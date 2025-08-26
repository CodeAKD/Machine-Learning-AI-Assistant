import streamlit as st
import os
import tempfile
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from nlp_pipeline import NLPPipeline
from vector_embeddings import VectorEmbeddings
import openai
from typing import List, Dict
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="📚 ML Books AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for awesome UI/UX with white, orange, mint, and lemon color scheme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom gradient background - White with subtle color hints */
    .stApp {
        background: linear-gradient(135deg, #ffffff 0%, #fefefe 25%, #fffef8 50%, #fffffb 75%, #ffffff 100%);
    }
    
    /* Main container styling */
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(255, 165, 0, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 165, 0, 0.1);
    }
    
    /* Header styling - Orange gradient text */
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #ff8c42 0%, #ffa726 50%, #ffb74d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling - White with mint accent */
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fffc 50%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(102, 187, 106, 0.1);
        transition: transform 0.3s ease;
        border: 1px solid rgba(102, 187, 106, 0.15);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(102, 187, 106, 0.15);
    }
    
    /* Chat container - Pure white with subtle shadow */
    .chat-container {
        background: #ffffff;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(255, 165, 0, 0.08);
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid rgba(255, 235, 59, 0.2);
    }
    
    /* Message styling - Orange gradient for user */
    .user-message {
        background: linear-gradient(135deg, #ff8c42 0%, #ffa726 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 5px 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(255, 140, 66, 0.3);
    }
    
    /* Assistant message - Mint gradient */
    .assistant-message {
        background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        font-weight: 500;
        box-shadow: 0 3px 10px rgba(102, 187, 106, 0.3);
    }
    
    /* Similarity score styling - Light lemon */
    .similarity-score {
        background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        font-weight: 600;
        text-align: center;
        color: #f57f17;
        border: 1px solid rgba(255, 235, 59, 0.3);
    }
    
    /* Source card styling - White with orange accent */
    .source-card {
        background: linear-gradient(135deg, #ffffff 0%, #fff8f0 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ff8c42;
        box-shadow: 0 3px 10px rgba(255, 140, 66, 0.1);
    }
    
    /* Upload area styling - White with lemon hint */
    .upload-area {
        border: 2px dashed #ffb74d;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #ffffff 0%, #fffef7 100%);
        margin: 1rem 0;
    }
    
    /* Button styling - Orange gradient */
    .stButton > button {
        background: linear-gradient(135deg, #ff8c42 0%, #ffa726 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 140, 66, 0.4);
        background: linear-gradient(135deg, #ff7043 0%, #ff9800 100%);
    }
    
    /* Sidebar styling - White with mint accent */
    .css-1d391kg {
        background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%);
    }
    
    /* Progress bar styling - Orange */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #ff8c42 0%, #ffa726 100%);
    }
    
    /* Metrics styling - Mint gradient */
    .metric-card {
        background: linear-gradient(135deg, #66bb6a 0%, #81c784 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 5px 15px rgba(102, 187, 106, 0.2);
    }
    
    /* Action button styling - Enhanced for main action buttons */
    .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #ff8c42 0%, #ffa726 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
        height: 80px;
        box-shadow: 0 8px 20px rgba(255, 140, 66, 0.3);
    }
    
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(255, 140, 66, 0.4);
        background: linear-gradient(135deg, #ff7043 0%, #ff9800 100%);
    }
    
    /* Section divider styling */
    .section-divider {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(135deg, #ff8c42 0%, #66bb6a 50%, #ffeb3b 100%);
        border-radius: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_embeddings' not in st.session_state:
    st.session_state.vector_embeddings = None
if 'nlp_pipeline' not in st.session_state:
    st.session_state.nlp_pipeline = None

@st.cache_resource
def load_nlp_pipeline():
    """Load NLP pipeline (cached)"""
    return NLPPipeline()

@st.cache_resource
def load_vector_embeddings():
    """Load vector embeddings (cached)"""
    ve = VectorEmbeddings()
    if os.path.exists('embeddings'):
        ve.load_embeddings('embeddings')
        return ve
    return None

def initialize_components():
    """Initialize NLP pipeline and vector embeddings"""
    if st.session_state.nlp_pipeline is None:
        with st.spinner("🔄 Loading NLP Pipeline..."):
            st.session_state.nlp_pipeline = load_nlp_pipeline()
    
    if st.session_state.vector_embeddings is None:
        with st.spinner("🔄 Loading Vector Embeddings..."):
            st.session_state.vector_embeddings = load_vector_embeddings()

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF file"""
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Create New_Entry directory if it doesn't exist
            new_entry_dir = "New_Entry"
            os.makedirs(new_entry_dir, exist_ok=True)
            
            # Process the PDF
            nlp = st.session_state.nlp_pipeline
            processed_data = nlp.process_pdf_file(tmp_file_path, new_entry_dir)
            
            if processed_data:
                # Add to vector embeddings
                ve = st.session_state.vector_embeddings
                if ve is not None:
                    metadata = {
                        'filename': uploaded_file.name,
                        'cleaned_filename': f"{Path(uploaded_file.name).stem}_cleaned.txt",
                        'file_path': os.path.join(new_entry_dir, f"{Path(uploaded_file.name).stem}_cleaned.txt"),
                        'content_length': len(processed_data['processed_text_lemmatized']),
                        'upload_date': datetime.now().isoformat()
                    }
                    
                    ve.add_new_document(
                        processed_data['processed_text_lemmatized'],
                        metadata
                    )
                    
                    return processed_data, metadata
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    return None, None

def generate_response(query, similar_docs):
    """
    Generate a RAG response using OpenAI API with citations from similar documents
    """
    if not similar_docs:
        return "I couldn't find any relevant documents for your query. Please try rephrasing your question or upload more documents."
    
    # Load environment variables first
    load_dotenv()
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        # Fallback to simple response if no API key
        doc_names = [doc['metadata']['filename'] for doc in similar_docs[:3]]
        response = f"Based on your query about '{query}', I found relevant information in the following documents:\n\n"
        for i, doc_name in enumerate(doc_names, 1):
            response += f"{i}. {doc_name}\n"
        response += "\n⚠️ Note: To get AI-generated responses with citations, please set your OPENAI_API_KEY in the .env file."
        return response
    
    try:
        # Get content from similar documents
        context_parts = []
        citations = []
        
        for i, doc in enumerate(similar_docs[:3]):  # Use top 3 most similar documents
            metadata = doc['metadata']
            filename = metadata['filename']
            
            # Read the actual content from the cleaned file
            try:
                with open(metadata['file_path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Take first 1000 characters to avoid token limits
                    content_snippet = content[:1000] + "..." if len(content) > 1000 else content
                    context_parts.append(f"From {filename}:\n{content_snippet}")
                    citations.append(f"[{i+1}] {filename} (Similarity: {doc['similarity_score']:.3f})")
            except Exception as e:
                st.error(f"Error reading content from {filename}: {str(e)}")
                continue
        
        # Combine context
        context = "\n\n".join(context_parts)
        
        # Create the prompt for OpenAI
        prompt = f"""You are a helpful AI assistant that answers questions based on provided documents. 
        
Context from documents:
{context}

User Question: {query}

Please provide a comprehensive answer based on the provided context. When referencing information from the documents, use citations like [1], [2], [3] to indicate which document the information comes from. Be specific and detailed in your response.

Answer:"""
        
        # Call OpenAI API using the new client format
        from openai import OpenAI
        client = OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that provides concise, well-structured answers in exactly 2 paragraphs with proper citations. Keep responses focused and under 300 tokens."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Add citations at the end
        full_response = ai_response + "\n\n**Sources:**\n" + "\n".join(citations)
        
        return full_response
        
    except Exception as e:
        st.error(f"Error generating AI response: {str(e)}")
        # Fallback to simple response
        doc_names = [doc['metadata']['filename'] for doc in similar_docs[:3]]
        response = f"Based on your query about '{query}', I found relevant information in the following documents:\n\n"
        for i, doc_name in enumerate(doc_names, 1):
            response += f"{i}. {doc_name}\n"
        response += "\nNote: AI response generation failed. Please check your OpenAI API configuration."
        return response

def main():
    # Header
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">📚 ML Books AI Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your intelligent companion for Machine Learning literature</p>', unsafe_allow_html=True)
    
    # Initialize components
    initialize_components()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Statistics
        if st.session_state.vector_embeddings:
            stats = st.session_state.vector_embeddings.get_document_stats()
            st.markdown(f'<div class="metric-card">📊 Total Documents<br><strong>{stats.get("total_documents", 0)}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🔍 Embedding Dimension<br><strong>{stats.get("embedding_dimension", 0)}</strong></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Number of similar documents", 1, 10, 5)
        show_scores = st.checkbox("Show similarity scores", True)
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        if st.button("🔄 Refresh Embeddings"):
            st.session_state.vector_embeddings = load_vector_embeddings()
            st.success("Embeddings refreshed!")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
    
    # Main content area - Dynamic responsive boxes
    st.markdown("### 🎯 Choose Your Action")
    
    # Create two main action boxes
    col1, col2 = st.columns([1, 1])
    
    # Initialize session state for selected action
    if 'selected_action' not in st.session_state:
        st.session_state.selected_action = None
    
    with col1:
        if st.button("📤 Upload PDF", key="upload_action", use_container_width=True):
            st.session_state.selected_action = "upload"
            st.rerun()
    
    with col2:
        if st.button("💬 AI Chat", key="chat_action", use_container_width=True):
            st.session_state.selected_action = "chat"
            st.rerun()
    
    # Dynamic content based on selected action
    if st.session_state.selected_action == "upload":
        st.markdown("---")
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload New PDF")
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to add to the knowledge base"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF", key="process_pdf"):
                with st.spinner("Processing PDF..."):
                    processed_data, metadata = process_uploaded_pdf(uploaded_file)
                    
                    if processed_data:
                        st.success(f"✅ Successfully processed: {uploaded_file.name}")
                        
                        # Show processing results
                        st.markdown("#### 📊 Processing Results")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Original Length", f"{len(processed_data['original_text'])} chars")
                        with col_b:
                            st.metric("Cleaned Length", f"{len(processed_data['processed_text_lemmatized'])} chars")
                        with col_c:
                            st.metric("Tokens", len(processed_data['lemmatized_tokens']))
                        
                        # Find similar documents
                        if st.session_state.vector_embeddings:
                            similar_docs = st.session_state.vector_embeddings.search_similar_documents(
                                processed_data['processed_text_lemmatized'][:500],  # Use first 500 chars
                                top_k=5
                            )
                            
                            if similar_docs:
                                st.markdown("#### 🔍 Similar Documents Found")
                                for doc in similar_docs:
                                    score = doc['similarity_score']
                                    filename = doc['metadata']['filename']
                                    
                                    st.markdown(f"""
                                    <div class="source-card">
                                        <strong>{filename}</strong>
                                        <div class="similarity-score">Similarity: {score:.3f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed to process PDF")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.selected_action == "chat":
        st.markdown("---")
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("### 💬 AI Chat Assistant")
        
        # Chat interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_input(
            "Ask me anything about ML books...",
            placeholder="e.g., What are the best algorithms for classification?",
            key="chat_input"
        )
        
        if st.button("🚀 Send", key="send_message") and user_query:
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_query
            })
            
            # Search for similar documents
            if st.session_state.vector_embeddings:
                with st.spinner("🔍 Searching knowledge base..."):
                    similar_docs = st.session_state.vector_embeddings.search_similar_documents(
                        user_query,
                        top_k=top_k
                    )
                    
                    # Generate response
                    response = generate_response(user_query, similar_docs)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    # Rerun to update chat display
                    st.rerun()
            else:
                st.error("❌ Vector embeddings not loaded. Please ensure the system is properly initialized.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Search Results Section
    if st.session_state.chat_history and st.session_state.vector_embeddings:
        st.markdown("---")
        st.markdown("### 🔍 Document Sources & Similarity Scores")
        
        # Get last user query
        last_query = None
        for message in reversed(st.session_state.chat_history):
            if message['role'] == 'user':
                last_query = message['content']
                break
        
        if last_query:
            similar_docs = st.session_state.vector_embeddings.search_similar_documents(
                last_query,
                top_k=top_k
            )
            
            if similar_docs:
                # Create columns for top 5 documents
                cols = st.columns(min(5, len(similar_docs)))
                
                for i, doc in enumerate(similar_docs[:5]):
                    with cols[i]:
                        score = doc['similarity_score']
                        filename = doc['metadata']['filename']
                        
                        # Color code based on similarity score
                        if score > 0.8:
                            color = "#4CAF50"  # Green
                        elif score > 0.6:
                            color = "#FF9800"  # Orange
                        else:
                            color = "#F44336"  # Red
                        
                        # Create clickable link to original PDF
                        pdf_path = f"ML Books/{filename.replace('_cleaned.txt', '.pdf')}"
                        pdf_file_path = os.path.join(os.getcwd(), pdf_path)
                        
                        # Check if PDF file exists
                        if os.path.exists(pdf_file_path):
                            # Create a downloadable link
                            with open(pdf_file_path, "rb") as pdf_file:
                                pdf_bytes = pdf_file.read()
                            
                            st.markdown(f"""
                            <div class="source-card" style="border-left-color: {color}">
                                <h4>📄 {filename.replace('_cleaned.txt', '.pdf')}</h4>
                                <div class="similarity-score" style="background: linear-gradient(135deg, {color}20, {color}40)">
                                    Similarity: {score:.3f}
                                </div>
                                <small>Rank: #{doc['rank']}</small>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Add download button for the PDF
                            st.download_button(
                                label="📖 Open PDF",
                                data=pdf_bytes,
                                file_name=filename.replace('_cleaned.txt', '.pdf'),
                                mime="application/pdf",
                                key=f"pdf_{i}"
                            )
                        else:
                            st.markdown(f"""
                            <div class="source-card" style="border-left-color: {color}">
                                <h4>📄 {filename}</h4>
                                <div class="similarity-score" style="background: linear-gradient(135deg, {color}20, {color}40)">
                                    Similarity: {score:.3f}
                                </div>
                                <small>Rank: #{doc['rank']}</small>
                                <small style="color: red;">⚠️ Original PDF not found</small>
                            </div>
                            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()