import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import List, Dict, Any, Optional

# Load environment variables
load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
        data_dir: str = "data",
        groq_api_key: str = None,
        vector_store_manager=None  
    ):
        print(f"[RAGSearch] Initializing...")

        # Get API key
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment or parameters")
        self.api_key = api_key

        self.llm_model = llm_model
        self.vector_store_manager = vector_store_manager
        
        print(f"[RAGSearch] Initialization complete")
        if vector_store_manager:
            print(f"[RAGSearch] Vector store manager type: {type(vector_store_manager)}")
    
    def _format_source_info(self, result, index: int) -> Dict[str, Any]:
         """Format source information based on content type"""
        metadata = result.metadata
    
        if metadata.get('source_type') == 'video':
        # Get video name safely
            video_name = metadata.get('file_name', 'Unknown Video')
            source_text = f"Video: {video_name}"
        
        # Handle duration safely - it might be None
            duration = metadata.get('duration')
            if duration is not None:
                try:
                # Convert to float if it's a string
                    if isinstance(duration, str):
                        duration = float(duration)
                    source_text += f" ({duration:.2f}s)"
                except (ValueError, TypeError):
                # If can't format as float, just show raw value
                    source_text += f" (Duration: {duration})"
        
        # Add transcript quality info
            has_transcript = metadata.get('has_transcript', False)
            transcript_source = metadata.get('transcript_source', 'unknown')
        
            return {
                'source_type': 'video',
                'metadata': metadata,
                'citation': f"[Source {index+1}: {source_text}]",
                'content_preview': result.page_content[:200] + "..." if len(result.page_content) > 200 else result.page_content,
                'has_transcript': has_transcript,
                'transcript_source': transcript_source,
                'duration': duration  # Store the raw duration
            }
        else:
        # Document source
            doc_name = metadata.get('file_name', 'Unknown Document')
            source_text = f"Document: {doc_name}"
        
             return {
                'source_type': 'document',
                'metadata': metadata,
                'citation': f"[Source {index+1}: {source_text}]",
                'content_preview': result.page_content[:200] + "..." if len(result.page_content) > 200 else result.page_content
            }
    
    def _create_context_prompt(self, context_parts: List[str], source_overview: str, 
                              sources_info: List[Dict], query: str) -> str:
        """Create the LLM prompt with context"""
        
        # Check if we have videos without transcripts
        videos_without_transcript = [s for s in sources_info if s.get('source_type') == 'video' and not s.get('has_transcript')]
        
        docs_context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

AVAILABLE SOURCES OVERVIEW:
{source_overview}

CONTEXT FROM RELEVANT SOURCES:
{docs_context}

QUESTION: {query}"""

        # Add warnings for videos without transcripts
        if videos_without_transcript:
            prompt += "\n\nIMPORTANT NOTES ABOUT SOURCES:"
            for source in videos_without_transcript:
                prompt += f"\n- {source['citation']} has NO transcript available. Information is very limited."
        
        prompt += """

INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer that question."
3. Reference sources like [Source 1], [Source 2], etc.
4. If a source has limited/no transcript, acknowledge this limitation
5. Be concise and accurate

ANSWER (based only on context, with source citations):"""
        
        return prompt
    
    def search(self, query: str, top_k: int = 3, filter_source: Optional[str] = None) -> Dict[str, Any]:
        try:
            print(f"[RAGSearch] Searching for: '{query}'")
            
            if self.vector_store_manager is None:
                return {
                    "answer": "No knowledge base available. Please add documents or videos first.",
                    "sources": [],
                    "total_sources": 0,
                    "context_used": False,
                    "query": query
                }
            
            # Query vector store
            results = self.vector_store_manager.query(
                query, 
                top_k=top_k,
                filter_source=filter_source
            )
            
            print(f"[RAGSearch] Found {len(results)} relevant chunks")
            
            if results and len(results) > 0:
                # Format context and sources
                context_parts = []
                sources_info = []
                
                for i, result in enumerate(results):
                    # Format source info
                    source_info = self._format_source_info(result, i)
                    sources_info.append(source_info)
                    
                    # Add to context
                    context_parts.append(f"{source_info['citation']}\n{result.page_content}")
                
                # Get source stats
                stats = self.vector_store_manager.get_source_stats()
                source_overview = f"Total: {stats['document_count']} documents, {stats['video_count']} videos"
                
                # Create prompt
                prompt = self._create_context_prompt(context_parts, source_overview, sources_info, query)
                
            else:
                # No results found
                sources_info = []
                stats = self.vector_store_manager.get_source_stats()
                
                prompt = f"""I searched the available sources but couldn't find specific information about: {query}

Available sources: {stats['document_count']} documents, {stats['video_count']} videos.

Please try:
1. Asking about something else in the loaded content
2. Rephrasing your question
3. Adding more documents or videos"""
            
            # Call LLM
            print(f"[RAGSearch] Calling LLM...")
            llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.llm_model,
                temperature=0.1,
                max_tokens=1024
            )
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer.strip(),
                "sources": sources_info,
                "total_sources": len(sources_info),
                "query": query,
                "context_used": len(results) > 0
            }
            
        except Exception as e:
            print(f"[RAGSearch ERROR] {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "total_sources": 0,
                "query": query,
                "context_used": False
            }
