from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Any, Union, Dict

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Loaded SentenceTransformer model: {model_name}")
        except Exception as e:
            print(f"Error loading SentenceTransformer model: {e}")
            raise

    def chunk_documents(self, documents: Union[List[Any], List[Dict]]) -> List[Dict]:
        """Chunk documents into smaller pieces with metadata preservation."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        all_chunks = []

        for doc in documents:
            # Handle both dict format and LangChain Document format
            if isinstance(doc, dict):
                content = doc.get("page_content", "")
                metadata = doc.get("metadata", {})
            else:
                # LangChain Document object
                content = getattr(doc, "page_content", None)
                metadata = getattr(doc, "metadata", {})
                if content is None:
                    print("Warning: Document missing 'page_content'. Skipping.")
                    continue

            if not content:
                print("Warning: Document missing 'page_content'. Skipping.")
                continue

            # Split the textm
            text_chunks = text_splitter.split_text(content)

            # Create chunk entries with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                })

                all_chunks.append({
                    'page_content': chunk_text,
                    'metadata': chunk_metadata
                })

        print(f"Chunked documents into {len(all_chunks)} pieces.")
        return all_chunks

    def embed_chunks(self, chunks: List[Dict], batch_size: int = 32) -> Dict[str, Any]:
      
        try:
            if not chunks:
                print("[WARNING] No chunks to embed")
                return {
                    'embeddings': np.array([]),
                    'chunks': []
                }

            print(f"[INFO] Embedding {len(chunks)} chunks...")

            # Extract text from chunks
            texts = [chunk['page_content'] for chunk in chunks]

            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=batch_size)

            print(f"[INFO] Generated embeddings shape: {embeddings.shape}")

            return {
                'embeddings': embeddings,
                'chunks': chunks  # Keep chunks with metadata
            }

        except Exception as e:
            print(f"[ERROR] Error in embedding chunks: {e}")
            raise
