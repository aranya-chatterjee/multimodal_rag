import streamlit as st
import tempfile
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import modules
from data_loader import load_documents as load_documents_files
from video_loader import VideoProcessor
from vectorStore import VectorStoreManager
from search import RAGSearch

st.set_page_config(
    page_title="VideoRAG - Local Video AI Assistant",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
def init_session_state():
    defaults = {
        "rag_search": None,
        "chats": [],
        "processed_sources": [],
        "processing_error": None,
        "vector_manager": None,
        "temp_dir": None,
        "whisper_model": "base",
        "processing_progress": 0,
        "current_process": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def clear_session():
    """Clear the session state."""
    keys_to_clear = ["rag_search", "chats", "processed_sources", 
                    "processing_error", "vector_manager", "temp_dir",
                    "processing_progress", "current_process"]
    
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = None if key.endswith("_dir") else (
                "base" if key == "whisper_model" else 
                0 if key == "processing_progress" else
                [] if key in ["chats", "processed_sources"] else 
                None
            )
    
    # Clean up temp directory
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        import shutil
        try:
            shutil.rmtree(st.session_state.temp_dir)
        except:
            pass
    
    st.rerun()

def update_progress(progress: int, message: str):
    """Update processing progress"""
    st.session_state.processing_progress = progress
    st.session_state.current_process = message

def process_document(uploaded_file):
    """Process uploaded document file"""
    try:
        st.info(f"📄 Processing file: {uploaded_file.name}")
        update_progress(10, "Saving file...")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="videorag_")
        st.session_state.temp_dir = temp_dir
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        update_progress(20, "File saved")
        time.sleep(0.5)

        # Load documents
        update_progress(30, "Loading document content...")
        docs = load_documents_files([file_path], source_type="files")
        
        if not docs or len(docs) == 0:
            st.session_state.processing_error = "No valid text extracted from the document!"
            update_progress(0, "Failed")
            return None, None, temp_dir
        
        update_progress(40, f"Loaded {len(docs)} document(s)")
        
        # Create/update vector store
        update_progress(50, "Creating embeddings...")
        if st.session_state.vector_manager is None:
            vector_manager = VectorStoreManager(
                persist_path=os.path.join(temp_dir, "faiss_store"),
                embed_model="all-MiniLM-L6-v2"
            )
            st.session_state.vector_manager = vector_manager
        else:
            vector_manager = st.session_state.vector_manager
        
        vector_manager.add_documents(docs, source_type="document")
        
        update_progress(70, "Embeddings created")

        # Initialize RAGSearch if not already
        if st.session_state.rag_search is None:
            try:
                api_key = st.secrets.get("GROQ_API_KEY", None)
            except (FileNotFoundError, AttributeError):
                api_key = None
            if not api_key:
                st.session_state.processing_error = "GROQ_API_KEY not found! Set it in Streamlit secrets."
                update_progress(0, "Failed")
                return None, None, temp_dir
            
            update_progress(80, "Initializing AI system...")
            rag_search = RAGSearch(
                persist_dir=os.path.join(temp_dir, "faiss_store"),
                data_dir=data_dir,
                embedding_model="all-MiniLM-L6-v2",
                llm_model="llama-3.1-8b-instant",
                groq_api_key=api_key,
                vector_store_manager=vector_manager
            )
            st.session_state.rag_search = rag_search
        
        # Track processed source
        source_info = {
            "type": "document",
            "name": uploaded_file.name,
            "file_type": uploaded_file.type,
            "timestamp": time.time()
        }
        st.session_state.processed_sources.append(source_info)
        
        update_progress(100, "Complete!")
        time.sleep(0.5)
        update_progress(0, None)
        
        return vector_manager, uploaded_file.name, temp_dir

    except Exception as e:
        st.session_state.processing_error = str(e)
        update_progress(0, "Error")
        return None, None, None

def process_video_file(uploaded_video):
    """Process uploaded video file"""
    try:
        st.info(f"🎬 Processing video: {uploaded_video.name}")
        update_progress(5, "Saving video...")
        
        # Create temporary directory
        if st.session_state.temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="videorag_")
            st.session_state.temp_dir = temp_dir
        else:
            temp_dir = st.session_state.temp_dir
        
        # Save uploaded video
        video_dir = os.path.join(temp_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        video_path = os.path.join(video_dir, uploaded_video.name)
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.getbuffer())
        
        update_progress(10, "Video saved")
        time.sleep(0.5)
        
        # Get video size for estimation
        video_size_mb = uploaded_video.size / (1024 * 1024)
        st.info(f"📊 Video size: {video_size_mb:.1f} MB")
        
        # Initialize video processor
        update_progress(15, "Initializing video processor...")
        video_processor = VideoProcessor(
            whisper_model=st.session_state.whisper_model,
            language="english"
        )
        
        # Estimate processing time (rough estimate: 10 seconds per MB for base model)
        estimated_time = int(video_size_mb * 10)
        if estimated_time > 60:
            time_text = f"{estimated_time//60} minutes"
        else:
            time_text = f"{estimated_time} seconds"
        
        st.info(f"⏱️ Estimated processing time: {time_text}")
        
        # Process video with progress updates
        def process_with_progress():
            update_progress(20, "Extracting audio from video...")
            result = video_processor.process_video_file(video_path)
            return result
        
        # Process video
        with st.spinner("🤖 Transcribing video..."):
            documents, video_info = process_with_progress()
        
        if not documents or not video_info:
            st.session_state.processing_error = "Failed to process video!"
            update_progress(0, "Failed")
            return None, None, temp_dir
        
        update_progress(60, "Transcription complete")
        
        # Convert documents to LangChain format
        from langchain_core.documents import Document
        langchain_docs = []
        for doc_dict in documents:
            langchain_docs.append(Document(
                page_content=doc_dict['page_content'],
                metadata=doc_dict['metadata']
            ))
        
        # Show video info
        if video_info.has_transcript:
            word_count = len(video_info.transcript_text.split())
            st.success(f"✅ Generated transcript: {word_count} words")
        else:
            st.warning("⚠️ No transcript generated - video might not have audio")
        
        # Create/update vector store
        update_progress(70, "Creating embeddings...")
        if st.session_state.vector_manager is None:
            vector_manager = VectorStoreManager(
                persist_path=os.path.join(temp_dir, "faiss_store"),
                embed_model="all-MiniLM-L6-v2"
            )
            st.session_state.vector_manager = vector_manager
        else:
            vector_manager = st.session_state.vector_manager
        
        vector_manager.add_content(langchain_docs, content_type="video")
        
        update_progress(85, "Embeddings created")

        # Initialize RAGSearch if not already
        if st.session_state.rag_search is None:
            try:
                api_key = st.secrets.get("GROQ_API_KEY", None)
            except (FileNotFoundError, AttributeError):
                api_key = None
            if not api_key:
                st.session_state.processing_error = "GROQ_API_KEY not found! Set it in Streamlit secrets."
                update_progress(0, "Failed")
                return None, None, temp_dir
            
            update_progress(90, "Initializing AI system...")
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            
            rag_search = RAGSearch(
                persist_dir=os.path.join(temp_dir, "faiss_store"),
                data_dir=data_dir,
                embedding_model="all-MiniLM-L6-v2",
                llm_model="llama-3.1-8b-instant",
                groq_api_key=api_key,
                vector_store_manager=vector_manager
            )
            st.session_state.rag_search = rag_search
        
        # Track processed source
        source_info = {
            "type": "video",
            "name": video_info.filename,
            "filesize": video_info.filesize,
            "duration": video_info.duration,
            "has_transcript": video_info.has_transcript,
            "transcript_source": video_info.transcript_source.value if video_info.transcript_source else None,
            "timestamp": time.time()
        }
        st.session_state.processed_sources.append(source_info)
        
        update_progress(100, "Complete!")
        time.sleep(0.5)
        update_progress(0, None)
        
        return vector_manager, video_info.filename, temp_dir

    except Exception as e:
        st.session_state.processing_error = str(e)
        update_progress(0, "Error")
        return None, None, None

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .small-text {
        font-size: 0.8em;
        color: #666;
    }
    .video-card {
        padding: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🎬 VideoRAG")
    st.markdown("---")
    
    # API Key status
    try:
        api_key = st.secrets.get("GROQ_API_KEY", None)
    except (FileNotFoundError, AttributeError):
        api_key = None
    
    if api_key:
        st.success(f"✅ API Key: Found ({api_key[:10]}...)")
    else:
        st.error("❌ API Key: Missing")
        with st.expander("How to add API key"):
            st.info("""
            1. Get a free API key from [console.groq.com](https://console.groq.com)
            2. Create a `.streamlit/secrets.toml` file
            3. Add this line:
            ```
            GROQ_API_KEY = "your-key-here"
            ```
            """)
    
    st.markdown("---")
    
    # Content type selection
    source_type = st.radio(
        "Choose content type:",
        ["📄 Document", "🎬 Video"],
        help="Upload documents or videos to create a knowledge base"
    )
    
    if source_type == "📄 Document":
        uploaded_file = st.file_uploader(
            "Upload Document",
            type=['pdf', 'txt', 'csv', 'docx', 'xlsx', 'json'],
            help="Supported formats: PDF, TXT, CSV, DOCX, XLSX, JSON"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"📄 {uploaded_file.name}")
            with col2:
                size_mb = uploaded_file.size / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")
            
            if st.button("🚀 Add to Knowledge Base", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    vector_manager, filename, temp_dir = process_document(uploaded_file)
                    
                    if vector_manager:
                        st.success(f"✅ Added: {filename}")
                        st.balloons()
                    elif st.session_state.processing_error:
                        st.error(f"❌ Error: {st.session_state.processing_error}")
    
    else:  # Video
        # Video settings
        with st.expander("⚙️ Settings", expanded=True):
            whisper_model = st.selectbox(
                "Transcription Model",
                ["tiny", "base", "small", "medium", "large"],
                index=1,
                help="""
                - **tiny**: Fastest, least accurate
                - **base**: Good balance (recommended)
                - **small**: Better accuracy
                - **medium**: High accuracy
                - **large**: Best accuracy, slowest
                """
            )
            st.session_state.whisper_model = whisper_model
            
            st.caption("💡 Tip: Use 'tiny' or 'base' for quick testing")
        
        # Video upload
        uploaded_video = st.file_uploader(
            "Upload Video",
            type=['mp4', 'mov', 'avi', 'mkv', 'wmv'],
            help="Supported formats: MP4, MOV, AVI, MKV, WMV"
        )
        
        if uploaded_video:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"🎬 {uploaded_video.name}")
            with col2:
                size_mb = uploaded_video.size / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")
            
            if uploaded_video.size > 100 * 1024 * 1024:  # 100 MB
                st.warning("⚠️ Large video may take a while to process")
            
            if st.button("🚀 Transcribe & Add", type="primary", use_container_width=True):
                # Show progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                if st.session_state.current_process:
                    status_text.text(st.session_state.current_process)
                    progress_bar.progress(st.session_state.processing_progress / 100)
                
                result = process_video_file(uploaded_video)
                
                if result[0]:  # vector_manager exists
                    vector_manager, filename, temp_dir = result
                    st.success(f"✅ Added: {filename}")
                    st.balloons()
                elif st.session_state.processing_error:
                    st.error(f"❌ Error: {st.session_state.processing_error}")
    
    st.markdown("---")
    
    # Current knowledge base
    if st.session_state.processed_sources:
        st.subheader("📊 Knowledge Base")
        
        # Stats
        if st.session_state.vector_manager:
            stats = st.session_state.vector_manager.get_source_stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total", stats['total_sources'])
            with col2:
                st.metric("Docs", stats['document_count'])
            with col3:
                st.metric("Videos", stats.get('video_count', 0))
        
        # Sources list
        with st.expander("📋 Sources", expanded=False):
            for source in st.session_state.processed_sources:
                if source["type"] == "document":
                    emoji = "📄"
                    info = f"{source['name']}"
                else:
                    emoji = "🎬"
                    duration = f" ({source['duration']:.1f}s)" if source.get('duration') else ""
                    transcript = "✅" if source.get('has_transcript') else "❌"
                    info = f"{source['name']}{duration} {transcript}"
                
                st.markdown(f"{emoji} {info}")
        
        # Clear button
        if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
            clear_session()
    
    st.markdown("---")
    st.caption("💡 Upload content and ask questions about it!")

# Main content area
st.title("🎬 VideoRAG - Local Video AI Assistant")
st.caption("Upload videos or documents, then ask questions about their content")

# Source filter
if st.session_state.rag_search and st.session_state.processed_sources:
    # Check if we have multiple source types
    source_types = set(s["type"] for s in st.session_state.processed_sources)
    
    if len(source_types) > 1:
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Ask a question:", placeholder="What would you like to know?", key="query_input")
        with col2:
            filter_option = st.selectbox(
                "Filter:",
                ["All sources", "Documents only", "Videos only"],
                key="source_filter"
            )
    else:
        query = st.text_input("Ask a question:", placeholder="What would you like to know?", key="query_input")
        filter_option = "All sources"
else:
    query = st.text_input("Ask a question:", placeholder="Add content first to ask questions", disabled=True)

# Chat interface
if st.session_state.rag_search and st.session_state.vector_manager:
    # Show loaded sources
    with st.expander("📁 Loaded Content", expanded=False):
        if st.session_state.processed_sources:
            for source in st.session_state.processed_sources:
                if source["type"] == "document":
                    st.markdown(f"📄 **{source['name']}**")
                else:
                    duration_text = f" • {source['duration']:.1f}s" if source.get('duration') else ""
                    transcript_status = "✅" if source.get('has_transcript') else "❌"
                    st.markdown(f"🎬 **{source['name']}**{duration_text} • Transcript: {transcript_status}")
        else:
            st.info("No content loaded yet")
    
    # Chat history
    if st.session_state.chats:
        st.subheader("💬 Conversation")
        
        for i, chat in enumerate(st.session_state.chats):
            with st.chat_message("user"):
                st.write(chat["user"])
            
            with st.chat_message("assistant", avatar="🤖"):
                st.write(chat["assistant"])
                
                # Show sources if available
                if chat.get("sources"):
                    with st.expander(f"📚 Sources ({len(chat['sources'])})", expanded=False):
                        for source in chat["sources"]:
                            if isinstance(source, dict):
                                if source.get('source_type') == 'video':
                                    emoji = "🎬"
                                    name = source['metadata'].get('file_name', 'Unknown Video')
                                else:
                                    emoji = "📄"
                                    name = source['metadata'].get('file_name', 'Unknown Document')
                                
                                st.markdown(f"{emoji} **{name}**")
                                if source.get('content_preview'):
                                    st.caption(f"*{source['content_preview']}*")
                            st.divider()
    
    # Handle query
    if query and st.button("🔍 Search", type="primary", use_container_width=True):
        # Add user message to chat
        st.session_state.chats.append({"user": query, "assistant": ""})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Determine filter for search
        filter_source = None
        if filter_option == "Documents only":
            filter_source = "document"
        elif filter_option == "Videos only":
            filter_source = "video"
        
        # Get assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG system
                    result = st.session_state.rag_search.search(query, top_k=3, filter_source=filter_source)
                    
                    # Display answer
                    st.write(result["answer"])
                    
                    # Update chat history
                    st.session_state.chats[-1]["assistant"] = result["answer"]
                    st.session_state.chats[-1]["sources"] = result.get("sources", [])
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chats[-1]["assistant"] = error_msg
    
    # Clear chat button
    if st.session_state.chats:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chats = []
            st.rerun()

else:
    # Welcome/Instructions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 👋 Welcome to VideoRAG!
        
        **Your personal AI assistant that understands videos and documents.**
        
        ### 🚀 **How it works:**
        1. **Upload** a video or document in the sidebar
        2. **Wait** for processing (videos take longer)
        3. **Ask questions** about the content
        4. **Get answers** with source references
        
        ### 📁 **Supported Content:**
        - **Videos:** MP4, MOV, AVI, MKV (with audio)
        - **Documents:** PDF, TXT, DOCX, XLSX, CSV, JSON
        
        ### 💡 **Tips for best results:**
        - Start with a **short video** (1-5 minutes) to test
        - For videos, ensure they have **clear audio**
        - Ask **specific questions** about the content
        - Larger videos may take **several minutes** to process
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 **Quick Start**
        
        **Try these test files:**
        
        **Documents:**
        - Meeting notes (TXT)
        - Research paper (PDF)
        - Report (DOCX)
        
        **Videos:**
        - Lecture recording
        - Meeting recording
        - Tutorial video
        - Presentation
        
        ### ⚡ **Quick Tips**
        - Use 'tiny' Whisper model for fast testing
        - Documents process instantly
        - Chat history is saved in session
        """)
    
    # Show previous error if any
    if st.session_state.processing_error:
        st.error(f"⚠️ Last error: {st.session_state.processing_error}")
        if st.button("🔄 Clear Error"):
            st.session_state.processing_error = None
            st.rerun()

# Footer
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])
with footer_col1:
    st.caption("🎬 Powered by VideoRAG | 🤖 Whisper AI + Groq LLM | 🛠️ Built with Streamlit")
with footer_col2:
    if st.session_state.chats:
        st.caption(f"💬 {len(st.session_state.chats)} messages")
with footer_col3:
    if st.session_state.processed_sources:
        st.caption(f"📁 {len(st.session_state.processed_sources)} sources")

# Save button in footer
if st.session_state.vector_manager and st.button("💾 Save Session", key="save_session"):
    st.session_state.vector_manager.save()
    st.success("Session saved successfully!")