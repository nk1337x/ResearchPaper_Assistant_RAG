"""
Retriever
Handles query processing and context retrieval from vector store.
"""

from typing import List, Dict, Tuple
import numpy as np
from src.embedder import EmbeddingGenerator
from src.vectorstore import VectorStore


class Retriever:
    """
    Retrieves relevant context from the vector store based on user queries.
    """
    
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingGenerator, 
                 top_k: int = 5, similarity_threshold: float = 0.3):
        """
        Initialize the retriever.
        
        Args:
            vector_store: VectorStore instance
            embedder: EmbeddingGenerator instance
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score to include result
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def retrieve(self, query: str) -> List[Dict[str, any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query string
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        # Generate embedding for query
        query_embedding = self.embedder.generate_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=self.top_k)
        
        # Filter by similarity threshold
        filtered_results = [
            r for r in results 
            if r['similarity_score'] >= self.similarity_threshold
        ]
        
        return filtered_results
    
    def format_context(self, retrieved_chunks: List[Dict[str, any]]) -> str:
        """
        Format retrieved chunks into a context string for the LLM.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant context found."
        
        context_parts = []
        
        for idx, result in enumerate(retrieved_chunks, 1):
            chunk = result['chunk']
            metadata = chunk['metadata']
            text = chunk['text']
            score = result['similarity_score']
            
            context_part = f"""
[Context {idx}] (Relevance: {score:.2f})
Paper: {metadata['paper_title']}
Section: {metadata['section']}
Content: {text}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def get_context_with_citations(self, retrieved_chunks: List[Dict[str, any]]) -> Tuple[str, List[str]]:
        """
        Format context and extract citations.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Tuple of (formatted_context, list_of_citations)
        """
        context = self.format_context(retrieved_chunks)
        
        # Extract unique paper references
        citations = []
        seen_papers = set()
        
        for result in retrieved_chunks:
            metadata = result['chunk']['metadata']
            paper_title = metadata['paper_title']
            section = metadata['section']
            
            if paper_title not in seen_papers:
                citations.append(f"{paper_title} ({section})")
                seen_papers.add(paper_title)
        
        return context, citations
