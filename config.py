"""
Configuration settings for Research Paper Assistant Chatbot.
"""

import os


class Config:
    """Configuration class for the chatbot."""
    
    # Embedding Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Chunking Parameters
    CHUNK_SIZE = 800  # characters
    CHUNK_OVERLAP = 200  # characters
    
    # Retrieval Parameters
    TOP_K = 5  # Number of chunks to retrieve
    SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score
    
    # Validation Parameters
    MIN_PAGES = 4
    MIN_SECTIONS = 3
    
    # Vector Store
    VECTOR_STORE_DIR = "data/vector_store"
    
    # Ollama Configuration (Local LLM)
    OLLAMA_MODEL = "mistral"  # Local LLM for research papers
    OLLAMA_BASE_URL = "http://localhost:11434"
    TEMPERATURE = 0.1  # Low temperature for more deterministic answers
    MAX_TOKENS = 500  # Maximum tokens in response
