import os 
from typing import List, Any, Dict, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import numpy as np 
import pickle 
from sentence_transformers import SentenceTransformer 
from ChunkAndEmbed import EmbeddingPipeline

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text])[0].tolist()

class VectorStoreManager:
    def __init__(self, persist_path: str="faiss_store", embed_model:str="all-MiniLM-L6-v2"):
        self.persist_path = persist_path
        self.embed_model = embed_model
        self.embedding_pipeline = EmbeddingPipeline(model_name=self.embed_model)
        self.vector_store = None
        self.loaded_sources = []  # Track loaded sources
        print(f"[VectorStore] Initialized")
    
    def add_content(self, documents: List[Any], content_type: str = "document", 
                   metadata: Optional[Dict] = None) -> bool:
        try:
            if not documents:
                print(f" No documents to add for {content_type}")
                return False

            print(f" First of the third part.. processing {content_type}...")
            chunks = self.embedding_pipeline.chunk_documents(documents)
            
            if not chunks:
                print("ERROR: No chunks created!")
                return False
            
            print(f"Created {len(chunks)} chunks")
            
            print(f"Second of the third part.. generating embeddings...")
            embedding_result = self.embedding_pipeline.embed_chunks(chunks)
            embeddings = embedding_result['embeddings']
            print(f"Embeddings shape: {embeddings.shape}")
            
            print(f"Third and the final part .. Adding to FAISS store...")
            
            # Create embeddings object
            embedding_function = SentenceTransformerEmbeddings(self.embedding_pipeline.model)
            
            # Prepare metadata
            metadatas = []
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict):
                    # Extract chunk metadata
                    chunk_metadata = chunk.get('metadata', {}).copy()
                    chunk_metadata.update({
                        'chunk_id': i,
                        'source_type': content_type,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })
                    # Add additional metadata if provided
                    if metadata:
                        chunk_metadata.update(metadata)
                    metadatas.append(chunk_metadata)
                else:
                    # Fallback for simple chunks
                    metadatas.append({
                        'chunk_id': i,
                        'source_type': content_type,
                        'chunk_index': i,
                        'total_chunks': len(chunks)
                    })
            
            # Extract texts
            if isinstance(chunks[0], dict):
                texts = [chunk['page_content'] for chunk in chunks]
            else:
                texts = chunks
            
            # Create or update FAISS store
            if self.vector_store is None:
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=embedding_function,
                    metadatas=metadatas
                )
            else:
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
            
            # Track source
            source_info = {
                'type': content_type,
                'chunk_count': len(chunks),
                'timestamp': np.datetime64('now')
            }
            
            # Add specific info based on content type
            if content_type == "video" and documents and hasattr(documents[0], 'metadata'):
                doc_metadata = documents[0].metadata
                source_info.update({
                    'filename': doc_metadata.get('file_name', 'Unknown'),
                    'duration': doc_metadata.get('duration', 0),
                    'transcript_source': doc_metadata.get('transcript_source', 'unknown'),
                    'has_transcript': doc_metadata.get('has_transcript', False)
                })
            elif content_type == "document" and documents and hasattr(documents[0], 'metadata'):
                doc_metadata = documents[0].metadata
                source_info.update({
                    'filename': doc_metadata.get('file_name', 'Unknown')
                })
            
            self.loaded_sources.append(source_info)
            
            print(f" Added {len(chunks)} {content_type} chunks to FAISS store")
            print(f" Total sources loaded: {len(self.loaded_sources)}")
            
            return True
            
        except Exception as e:
            print(f" Error adding content: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Backward compatibility methods
    def build_vector_store(self, documents: List[Any], source_type: str = "document"):
        return self.add_content(documents, source_type)
    
    def add_documents(self, documents: List[Any], source_type: str = "document"):
        return self.add_content(documents, source_type)
    
    def add_video(self, documents: List[Any]):
        return self.add_content(documents, "video")
    
    def _test_query(self, sample_text: str):
        # Test with actual content
        print("\n🔍 Testing vector store...")
        
        # Extract words from sample
        words = [w for w in sample_text.split() if len(w) > 3]
        if words:
            # Try the first word
            test_word = words[0]
            print(f"Testing query: '{test_word}'")
            
            results = self.query(test_word, top_k=1)
            print(f"Results found: {len(results)}")
            
            if results:
                print(" Vector store WORKS!")
                content = results[0].page_content
                print(f"Found: {content[:100]}...")
            else:
                print(" No results found")
    
    def query(self, query_text: str, top_k: int = 5, filter_source: Optional[str] = None) -> List[Any]:
       
        if not self.vector_store:
            print("No vector store!")
            return []
        
        try:
            if filter_source:
                # Get more results to filter
                results = self.vector_store.similarity_search(query_text, k=top_k * 3)
                
                # Filter by source type
                filtered_results = []
                for result in results:
                    metadata = result.metadata
                    if metadata.get('source_type') == filter_source:
                        filtered_results.append(result)
                        if len(filtered_results) >= top_k:
                            break
                
                print(f"Query '{query_text}' found {len(filtered_results)} results (filtered for {filter_source})")
                return filtered_results[:top_k]
            else:
                # Regular query
                results = self.vector_store.similarity_search(query_text, k=top_k)
                print(f"Query '{query_text}' found {len(results)} results")
                return results
                
        except Exception as e:
            print(f"Query error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_source_stats(self) -> Dict:
        """Get statistics about loaded sources"""
        stats = {
            'total_sources': len(self.loaded_sources),
            'document_count': 0,
            'video_count': 0,
            'total_chunks': 0
        }
        
        if self.vector_store:
            index_to_docstore_id = self.vector_store.index_to_docstore_id
            stats['total_chunks'] = len(index_to_docstore_id)
        
        # Count by source type
        for source in self.loaded_sources:
            if source['type'] == 'document':
                stats['document_count'] += 1
            elif source['type'] == 'video':
                stats['video_count'] += 1
        
        return stats
    
    def clear(self):
        """Clear all data from vector store"""
        self.vector_store = None
        self.loaded_sources = []
        print("Vector store cleared")
    
    def save(self):
        """Save vector store to disk"""
        if self.vector_store:
            self.vector_store.save_local(self.persist_path)
            
            # Also save source metadata
            metadata_path = os.path.join(self.persist_path, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.loaded_sources, f)
            
            print(f"Saved to {self.persist_path}")
    
    def load(self):
        """Load vector store from disk"""
        try:
            # Create Embeddings object for loading
            embedding_function = SentenceTransformerEmbeddings(self.embedding_pipeline.model)
            
            # Load FAISS store
            self.vector_store = FAISS.load_local(
                folder_path=self.persist_path,
                embeddings=embedding_function,  
                allow_dangerous_deserialization=True
            )
            
            # Load metadata
            metadata_path = os.path.join(self.persist_path, "metadata.pkl")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.loaded_sources = pickle.load(f)
            
            print(f"Loaded from {self.persist_path}")
            print(f"📊 Loaded {len(self.loaded_sources)} sources")
            return True
        except Exception as e:
            print(f"Load failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        # building a test script 
def create_test_documents():
    """Create test documents for vector store testing"""
    from langchain_core.documents import Document
    
    # Create document-style test data
    document_docs = [
        Document(
            page_content="This is a test document about artificial intelligence. AI is transforming many industries.",
            metadata={
                'source_type': 'document',
                'file_name': 'test_ai_doc.txt',
                'content_type': 'text'
            }
        ),
        Document(
            page_content="Machine learning is a subset of AI. It allows computers to learn from data.",
            metadata={
                'source_type': 'document', 
                'file_name': 'test_ml_doc.txt',
                'content_type': 'text'
            }
        )
    ]
    
    # Create video-style test data (simulated transcripts)
    video_docs = [
        Document(
            page_content="Welcome to this video about artificial intelligence. Today we'll discuss AI basics.",
            metadata={
                'source_type': 'video',
                'file_name': 'ai_intro.mp4',
                'content_type': 'video',
                'has_transcript': True,
                'duration': 300
            }
        )
    ]
    
    return document_docs, video_docs

def run_quick_test():
    """Quick test to verify vector store functionality"""
    print("🧪 Running Quick Vector Store Test...")
    print("=" * 50)
    
    try:
        # Create test directory
        import tempfile
        test_dir = tempfile.mkdtemp(prefix="vector_test_")
        
        # Initialize vector store
        print("\n🔄 Initializing VectorStoreManager...")
        vector_store = VectorStoreManager(persist_path=os.path.join(test_dir, "faiss_store"))
        
        # Create test documents
        document_docs, video_docs = create_test_documents()
        
        # Test 1: Add documents
        print("\n1️⃣ Testing document addition...")
        success = vector_store.add_content(document_docs, content_type="document")
        
        if not success:
            print("❌ Failed to add documents")
            return False
        
        print(f"✅ Added {len(document_docs)} documents")
        
        # Test 2: Add videos
        print("\n2️⃣ Testing video addition...")
        success = vector_store.add_content(video_docs, content_type="video")
        
        if not success:
            print("❌ Failed to add videos")
            return False
        
        print(f"✅ Added {len(video_docs)} videos")
        
        # Test 3: Check stats
        print("\n3️⃣ Checking statistics...")
        stats = vector_store.get_source_stats()
        
        print(f"   Total sources: {stats['total_sources']}")
        print(f"   Document count: {stats['document_count']}")
        print(f"   Video count: {stats['video_count']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        
        # Test 4: Test queries
        print("\n4️⃣ Testing queries...")
        
        # Test general query
        print("   Testing query: 'artificial intelligence'")
        results = vector_store.query("artificial intelligence", top_k=2)
        print(f"   Found {len(results)} results")
        
        if results:
            print(f"   ✅ Query successful")
            print(f"   First result preview: {results[0].page_content[:80]}...")
        else:
            print(f"   ❌ No results found")
        
        # Test filtered query
        print("\n   Testing filtered query (documents only): 'machine learning'")
        results = vector_store.query("machine learning", top_k=2, filter_source="document")
        print(f"   Found {len(results)} document results")
        
        # Test 5: Save and load
        print("\n5️⃣ Testing save/load functionality...")
        save_success = vector_store.save()
        
        if save_success:
            print("   ✅ Save successful")
            
            # Create new instance and load
            new_vector_store = VectorStoreManager(persist_path=os.path.join(test_dir, "faiss_store"))
            load_success = new_vector_store.load()
            
            if load_success:
                print("   ✅ Load successful")
                new_stats = new_vector_store.get_source_stats()
                print(f"   Loaded stats match: {new_stats == stats}")
            else:
                print("   ❌ Load failed")
        else:
            print("   ❌ Save failed")
        
        # Test 6: Clear functionality
        print("\n6️⃣ Testing clear functionality...")
        vector_store.clear()
        stats_after_clear = vector_store.get_source_stats()
        
        if stats_after_clear['total_sources'] == 0:
            print("   ✅ Clear successful")
        else:
            print("   ❌ Clear failed")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        print(f"\n🧹 Cleaned up test directory")
        
        print("\n" + "=" * 50)
        print("🎉 VECTOR STORE TEST PASSED!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic vector store operations"""
    print("🔧 Testing Basic Vector Store Functionality")
    print("=" * 50)
    
    try:
        import tempfile
        test_dir = tempfile.mkdtemp()
        
        # Create vector store
        vs = VectorStoreManager(persist_path=os.path.join(test_dir, "test_store"))
        
        # Create a simple test document
        from langchain_core.documents import Document
        test_doc = Document(
            page_content="The quick brown fox jumps over the lazy dog.",
            metadata={'file_name': 'test.txt', 'source_type': 'document'}
        )
        
        # Add document
        print("\nAdding test document...")
        success = vs.add_content([test_doc], content_type="document")
        
        if not success:
            print("❌ Failed to add document")
            return False
        
        # Query
        print("\nQuerying vector store...")
        results = vs.query("quick brown fox", top_k=1)
        
        if results and len(results) > 0:
            print(f"✅ Query successful")
            print(f"Result: {results[0].page_content}")
            print(f"Metadata: {results[0].metadata}")
        else:
            print("❌ No results found")
            return False
        
        # Get stats
        stats = vs.get_source_stats()
        print(f"\n📊 Stats: {stats}")
        
        # Save
        vs.save()
        print("💾 Saved successfully")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def performance_test():
    """Test vector store performance with multiple documents"""
    print("⚡ Running Performance Test")
    print("=" * 50)
    
    import tempfile
    import time
    from langchain_core.documents import Document
    
    test_dir = tempfile.mkdtemp()
    
    try:
        vs = VectorStoreManager(persist_path=os.path.join(test_dir, "perf_store"))
        
        # Create multiple test documents
        documents = []
        for i in range(5):  # Create 5 test documents
            documents.append(Document(
                page_content=f"Document {i+1}: This is test content about topic {i+1}. " * 10,
                metadata={'file_name': f'doc_{i+1}.txt', 'source_type': 'document'}
            ))
        
        print(f"Created {len(documents)} test documents")
        
        # Time the addition
        start_time = time.time()
        success = vs.add_content(documents, content_type="document")
        add_time = time.time() - start_time
        
        if success:
            print(f"✅ Added {len(documents)} documents in {add_time:.2f} seconds")
            
            # Time queries
            queries = ["test content", "topic", "document"]
            for query in queries:
                start_time = time.time()
                results = vs.query(query, top_k=3)
                query_time = time.time() - start_time
                print(f"   Query '{query}': {len(results)} results in {query_time:.3f}s")
            
            # Check stats
            stats = vs.get_source_stats()
            print(f"\n📊 Final stats: {stats}")
            
        else:
            print("❌ Failed to add documents")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        return success
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

# ==============================================
# MAIN TEST RUNNER
# ==============================================

if __name__ == "__main__":
    """
    Run when vectorStore.py is executed directly
    """
    print("=" * 60)
    print("VECTOR STORE - Built-in Test Suite")
    print("=" * 60)
    
    # Check if running in interactive mode
    try:
        from IPython import get_ipython
        is_interactive = get_ipython() is not None
    except:
        is_interactive = False
    
    if not is_interactive:
        # Command line execution
        import sys
        
        if len(sys.argv) > 1:
            test_type = sys.argv[1].lower()
            
            if test_type == "quick":
                success = run_quick_test()
            elif test_type == "basic":
                success = test_basic_functionality()
            elif test_type == "performance":
                success = performance_test()
            else:
                print(f"Unknown test type: {test_type}")
                print("Available: quick, basic, performance")
                sys.exit(1)
            
            sys.exit(0 if success else 1)
        else:
            # Interactive menu
            print("\nAvailable tests:")
            print("1. Quick test (comprehensive)")
            print("2. Basic functionality test")
            print("3. Performance test")
            
            try:
                choice = input("\nEnter choice (1-3): ").strip()
                
                if choice == "1":
                    print("\nRunning quick test...")
                    success = run_quick_test()
                elif choice == "2":
                    print("\nRunning basic test...")
                    success = test_basic_functionality()
                elif choice == "3":
                    print("\nRunning performance test...")
                    success = performance_test()
                else:
                    print("\nRunning quick test by default...")
                    success = run_quick_test()
                
                sys.exit(0 if success else 1)
                
            except KeyboardInterrupt:
                print("\n\nTest cancelled by user")
                sys.exit(1)
    else:
        # Running in interactive mode (like Jupyter)
        print("\n📚 VectorStore module loaded successfully!")
        print("\nAvailable test functions:")
        print("  - run_quick_test() : Comprehensive test")
        print("  - test_basic_functionality() : Basic operations test")
        print("  - performance_test() : Performance test")
        print("\nExample usage:")
        print("  from vectorStore import VectorStoreManager, run_quick_test")
        print("  run_quick_test()  # Run comprehensive test")
        print("\nOr create and use vector store:")
        print("  vs = VectorStoreManager()")
        print("  vs.add_content(documents)")
        print("  results = vs.query('your question')")