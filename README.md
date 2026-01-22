# Multimodal RAG Application

A Streamlit-based multimodal RAG (Retrieval-Augmented Generation) application that processes documents and videos, creates embeddings, and provides AI-powered question answering using Groq LLM.

## Features

- 📄 **Document Processing**: Supports PDF, TXT, CSV, DOCX, XLSX, JSON
- 🎬 **Video Processing**: Supports MP4, MOV, AVI, MKV, WMV with audio transcription
- 🔍 **Vector Search**: FAISS-based vector store for semantic search
- 🤖 **AI Chat**: Groq LLM integration for intelligent Q&A
- 💾 **Session Management**: Save and load vector stores

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Groq API Key**
   
   Create a `.streamlit/secrets.toml` file in the project root:
   ```toml
   GROQ_API_KEY = "your-groq-api-key-here"
   ```
   
   Get your API key from [console.groq.com](https://console.groq.com)

3. **Run the Application**
   ```bash
   streamlit run src/app.py
   ```

## Usage

1. **Upload Content**: 
   - Choose "Document" or "Video" in the sidebar
   - Upload your file
   - Click "Add to Knowledge Base" or "Transcribe & Add"

2. **Ask Questions**:
   - Type your question in the main area
   - Click "Search" to get AI-powered answers
   - View source references for each answer

3. **Filter Sources**:
   - Use the filter dropdown to search only documents or videos

## Project Structure

```
multimodal_rag/
├── src/
│   ├── app.py              # Main Streamlit application
│   ├── data_loader.py      # Document loading utilities
│   ├── video_loader.py      # Video processing and transcription
│   ├── vectorStore.py      # FAISS vector store management
│   ├── search.py           # RAG search and LLM integration
│   └── ChunkAndEmbed.py    # Text chunking and embedding
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Notes

- Video processing uses Whisper AI for transcription (can take time for large videos)
- The vector store is stored temporarily during the session
- Use "Save Session" to persist the vector store to disk

## Troubleshooting

- **API Key Error**: Make sure `.streamlit/secrets.toml` exists with your GROQ_API_KEY
- **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Video Processing Slow**: Use smaller videos or the "tiny" Whisper model for testing
