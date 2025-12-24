"""
Vector Store Manager
Manages FAISS vector database for storing and retrieving embeddings.
"""

from typing import List, Dict, Tuple
import numpy as np
import faiss
import pickle
import os


class VectorStore:
    """
    Manages a FAISS vector database for efficient similarity search.
    """
    
    def __init__(self, embedding_dimension: int):
        """
        Initialize the vector store.
        
        Args:
            embedding_dimension: Dimension of the embeddings
        """
        self.embedding_dimension = embedding_dimension
        self.index = None
        self.chunks = []  # Store chunk data and metadata
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        # Using IndexFlatL2 for exact search (good for small to medium datasets)
        # For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        print(f"Initialized FAISS index with dimension {self.embedding_dimension}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, any]]):
        """
        Add embeddings and their associated chunks to the vector store.
        
        Args:
            embeddings: Numpy array of embeddings (shape: [num_vectors, dimension])
            chunks: List of chunk dictionaries with text and metadata
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store chunk data
        self.chunks.extend(chunks)
        
        print(f"Added {len(embeddings)} embeddings. Total vectors: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, any]]:
        """
        Search for similar vectors in the store.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of dictionaries containing chunks and their similarity scores
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D array and float32
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        k = min(k, self.index.ntotal)  # Can't return more than we have
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):  # Validity check
                result = {
                    "chunk": self.chunks[idx],
                    "distance": float(distance),
                    "similarity_score": float(1 / (1 + distance))  # Convert distance to similarity
                }
                results.append(result)
        
        return results
    
    def save(self, directory: str, index_name: str = "faiss_index"):
        """
        Save the vector store to disk.
        
        Args:
            directory: Directory to save the index
            index_name: Name for the index file (without extension)
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, f"{index_name}.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save chunks metadata
        chunks_path = os.path.join(directory, f"{index_name}_chunks.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"Saved vector store to {directory}")
    
    def load(self, directory: str, index_name: str = "faiss_index"):
        """
        Load the vector store from disk.
        
        Args:
            directory: Directory containing the index
            index_name: Name of the index file (without extension)
        """
        # Load FAISS index
        index_path = os.path.join(directory, f"{index_name}.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # Load chunks metadata
        chunks_path = os.path.join(directory, f"{index_name}_chunks.pkl")
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
        
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"Loaded vector store from {directory}")
        print(f"Total vectors: {self.index.ntotal}")
    
    def clear(self):
        """Clear all data from the vector store."""
        self._initialize_index()
        self.chunks = []
        print("Vector store cleared")
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embedding_dimension,
            "papers": len(set(chunk["metadata"]["file_name"] for chunk in self.chunks))
        }
