"""
Section-Aware Text Chunker
Chunks text from research papers while preserving section context.
"""

from typing import List, Dict
import re


class SectionAwareChunker:
    """
    Chunks text from research papers in a section-aware manner.
    Preserves semantic context by keeping section information.
    """
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_sections(self, sections: List[Dict[str, any]], 
                          metadata: Dict[str, str], 
                          file_name: str) -> List[Dict[str, any]]:
        """
        Chunk text while preserving section boundaries.
        
        Args:
            sections: List of section dictionaries from PDFExtractor
            metadata: Document metadata
            file_name: Name of the source file
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        chunk_id = 0
        
        for section in sections:
            section_title = section["title"]
            section_content = section["content"]
            
            # Skip very short sections
            if len(section_content.strip()) < 50:
                continue
            
            # If section is small enough, keep as one chunk
            if len(section_content) <= self.chunk_size:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": f"Section: {section_title}\n\n{section_content}",
                    "metadata": {
                        "paper_title": metadata.get("title", "Unknown"),
                        "section": section_title,
                        "file_name": file_name,
                        "author": metadata.get("author", "Unknown"),
                        "chunk_index": 0,
                        "total_chunks_in_section": 1
                    }
                })
                chunk_id += 1
            else:
                # Split large sections into multiple chunks
                section_chunks = self._split_section(section_content, section_title)
                
                for idx, chunk_text in enumerate(section_chunks):
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": f"Section: {section_title}\n\n{chunk_text}",
                        "metadata": {
                            "paper_title": metadata.get("title", "Unknown"),
                            "section": section_title,
                            "file_name": file_name,
                            "author": metadata.get("author", "Unknown"),
                            "chunk_index": idx,
                            "total_chunks_in_section": len(section_chunks)
                        }
                    })
                    chunk_id += 1
        
        return chunks
    
    def _split_section(self, text: str, section_title: str) -> List[str]:
        """
        Split a large section into smaller chunks with overlap.
        
        Args:
            text: Section text to split
            section_title: Name of the section
            
        Returns:
            List of text chunks
        """
        # Split by sentences to avoid breaking mid-sentence
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap (keep last few sentences)
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    # Keep only sentences that fit in overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add remaining text as final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Basic sentence splitting (handles most cases)
        # Split on . ! ? followed by space and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk_simple(self, text: str, metadata: Dict[str, str], 
                     file_name: str) -> List[Dict[str, any]]:
        """
        Simple chunking without section awareness (fallback method).
        
        Args:
            text: Full text to chunk
            metadata: Document metadata
            file_name: Name of the source file
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": ' '.join(current_chunk),
                    "metadata": {
                        "paper_title": metadata.get("title", "Unknown"),
                        "section": "Unknown",
                        "file_name": file_name,
                        "author": metadata.get("author", "Unknown"),
                        "chunk_index": chunk_id
                    }
                })
                chunk_id += 1
                
                # Handle overlap
                overlap_text = ' '.join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    current_chunk = current_chunk[-(self.chunk_overlap // 100):]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": ' '.join(current_chunk),
                "metadata": {
                    "paper_title": metadata.get("title", "Unknown"),
                    "section": "Unknown",
                    "file_name": file_name,
                    "author": metadata.get("author", "Unknown"),
                    "chunk_index": chunk_id
                }
            })
        
        return chunks
