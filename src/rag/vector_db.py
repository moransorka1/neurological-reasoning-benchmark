from typing import List, Dict, Any, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class VectorDatabase:
    """
    Handles vector database operations for RAG implementations.
    """
    def __init__(self, 
                 persist_directory: str,
                 embedding_model_path: str = None,
                 device: str = "cuda:0"):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Directory for the persistent vector database
            embedding_model_path: Path to embedding model
            device: Device to use for embeddings generation (cuda:0, cpu, etc.)
        """
        self.persist_directory = persist_directory
        self.embedding_model_path = embedding_model_path
        self.device = device
        self.vectorstore = None
        
        # Initialize embeddings
        if embedding_model_path:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_path,
                model_kwargs={
                    'device': device
                },
                encode_kwargs={
                    'normalize_embeddings': True,
                    'batch_size': 32
                }
            )
        else:
            # Use a default path or handle the case when no path is provided
            self.embeddings = None
            
        # Load or initialize the vector database
        self._load_vectorstore()
    
    def _load_vectorstore(self):
        """
        Load the vector database from the persist directory.
        """
        if self.embeddings:
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
            except Exception as e:
                print(f"Error loading vector database: {e}")
                self.vectorstore = None
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a similarity search on the vector database.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of document dictionaries with page_content and metadata
        """
        if not self.vectorstore:
            raise ValueError("Vector database not initialized")
            
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            })
            
        return formatted_results