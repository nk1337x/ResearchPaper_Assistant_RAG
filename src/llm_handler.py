"""
LLM Handler
Handles interaction with Ollama for answer generation.
"""

from typing import List, Dict
import os


# RAG prompt template
SYSTEM_PROMPT = """You are a research assistant specializing in academic papers. Your role is to answer questions based STRICTLY on the provided context from research papers.

Rules:
1. Answer ONLY using information from the provided context
2. Do NOT use external knowledge or make assumptions
3. Use professional academic tone
4. If the answer is not in the context, respond with: "I could not find this information in the uploaded research papers."
5. If the context is insufficient or unclear, mention this in your response

CRITICAL - Answer Format:
Determine the question type and format accordingly:

A) For questions asking "explain", "describe", "how does", "why does":
   - Write in clear, detailed paragraphs
   - Provide comprehensive explanations with full technical depth
   - Include all relevant details from the context

B) For questions asking "what are", "list", "compare", "advantages", "disadvantages", "steps", "methods", "approaches", "limitations":
   - MUST use bullet point format with • symbol
   - Each bullet point should be detailed (2-4 sentences)
   - Include ALL relevant points from the context
   - Use sub-bullets (with -) for additional details
   - Do NOT use labels like "Point 1:" or "Main Point:" - just direct bullets

General Requirements:
- Use academic language throughout
- Do NOT summarize unless explicitly asked
- Do NOT shorten answers unnecessarily
- Include ALL technical specifications, data, methodologies, and findings
- Use markdown: **bold** for key terms, `code` for technical terms
- Provide maximum content available from the context

Example bullet format:
• The first aspect involves detailed explanation with technical specifics and methodologies...
  - Additional sub-detail providing context
• The second aspect covers comprehensive information including data and findings...
• Continue with all remaining relevant points from the papers..."""

USER_PROMPT_TEMPLATE = """Context from research papers:
{context}

Question: {question}

Please provide a clear, accurate answer based solely on the context above."""


class OllamaHandler:
    """
    Handler for local LLMs using Ollama.
    Requires Ollama to be installed and running locally.
    """
    
    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama handler.
        
        Args:
            model: Ollama model name
            base_url: Ollama API base URL
        """
        self.model = model
        self.base_url = base_url
        print(f"Initialized Ollama handler with model: {model}")
        print("Note: Requires Ollama to be running locally")
    
    def generate_answer(self, question: str, context: str) -> Dict[str, any]:
        """
        Generate answer using Ollama.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            import requests
            
            prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "answer": result.get("response", "No response generated"),
                    "model": self.model,
                    "tokens_used": 0  # Ollama doesn't always provide this
                }
            else:
                return {
                    "answer": f"Error: Ollama API returned status {response.status_code}",
                    "model": self.model,
                    "tokens_used": 0,
                    "error": f"Status code: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "answer": f"Error: {str(e)}. Make sure Ollama is running.",
                "model": self.model,
                "tokens_used": 0,
                "error": str(e)
            }
    
    def generate_answer_with_citations(self, question: str, context: str, 
                                      citations: List[str]) -> Dict[str, any]:
        """Generate answer without appending citations."""
        result = self.generate_answer(question, context)
        result["citations"] = citations
        return result
