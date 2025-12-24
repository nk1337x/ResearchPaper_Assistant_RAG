# Research Paper Assistant Chatbot

A sophisticated **Retrieval-Augmented Generation (RAG)** chatbot that allows users to upload research papers and ask questions, receiving accurate answers based strictly on the uploaded content. The system validates that only legitimate research papers are accepted and rejects non-research PDFs.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 🔍 Research Paper Validation
- **Strict validation** to ensure only research papers are processed
- Validates PDF format, minimum page count (≥4 pages)
- Detects standard research sections (Abstract, Introduction, Methodology, Results, Conclusion, References)
- Identifies citation patterns ([1], et al., DOI)
- Rejects non-research documents with clear error messages

### 🎯 Advanced RAG Architecture
- **Section-aware chunking** preserves semantic context
- **Semantic search** using sentence-transformers embeddings
- **FAISS vector database** for efficient similarity search
- **Context-aware retrieval** with relevance scoring
- **LLM integration** with strict context adherence

### 🚫 No Hallucination
- Answers generated **strictly from uploaded papers**
- No external knowledge or assumptions
- Clear "information not found" responses when appropriate
- Citations with paper title and section references

### 💡 User-Friendly Interface
- Clean, intuitive Streamlit UI
- Real-time paper upload and validation
- Chat interface with conversation history
- Paper management and statistics
- Professional academic tone in responses

## 🏗️ System Architecture

```
User Upload → Validation → Text Extraction → Section Chunking
                ↓
            [REJECT: "Only research papers allowed"]
                ↓
         Embedding Generation → Vector Store (FAISS)
                ↓
         User Query → Retrieve Top-K → LLM → Answer
```

## 📋 Requirements

- Python 3.9 or higher
- Ollama with Mistral model (free local LLM)
- 4GB+ RAM recommended
- Internet connection (for downloading models on first run)

## 🚀 Quick Start

### 1. Clone and Navigate
```bash
cd Research_Paper_Chatbot
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama and Download Model

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Install and run Ollama
3. Pull the Mistral model:
```bash
ollama pull mistral
```

### 5. Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Uploading Papers
1. Click the **Upload Research Papers** button in the sidebar
2. Select one or more PDF research papers
3. The system validates each paper:
   - ✅ Valid research papers are processed and added
   - ❌ Non-research PDFs are rejected with an error message

### Asking Questions
1. Type your question in the input box
2. Click **Ask** or press Enter
3. The system will:
   - Search for relevant content across all uploaded papers
   - Generate an answer based strictly on the retrieved context
   - Cite the source paper and section

### Example Questions
- "What methodology was used in this study?"
- "What were the main findings?"
- "How does this paper define neural networks?"
- "What datasets were used for evaluation?"
- "What are the limitations mentioned?"

### Managing Papers
- View uploaded papers in the sidebar
- See paper statistics (sections, chunks)
- Clear all papers with the **Clear All Papers** button

## 🗂️ Project Structure

```
Research_Paper_Chatbot/
├── src/
│   ├── __init__.py
│   ├── validator.py          # Research paper validation logic
│   ├── extractor.py          # PDF text extraction
│   ├── chunker.py            # Section-aware chunking
│   ├── embedder.py           # Embedding generation
│   ├── vectorstore.py        # FAISS vector database
│   ├── retriever.py          # Context retrieval
│   └── llm_handler.py        # LLM integration (Ollama)
├── app.py                     # Streamlit UI application
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── ARCHITECTURE.md           # Detailed system architecture
```

## ⚙️ Configuration

Edit [config.py](config.py) to customize settings:

```python
# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 200       # characters

# Retrieval
TOP_K =5               # chunks to retrieve
SIMILARITY_THRESHOLD = 0.3

# LLM
LLM_MODEL = "gpt-3.5-turbo"  # OpenAI model (when USE_OLLAMA = False)
TEMPERATURE = 0.1
MAX_TOKENS = 500

# Ollama (Default - Free Local LLM)
USE_OLLAMA = True
OLLAMA_MODEL = "mistral"

# Validation
MIN_PAGES = 4
MIN_SECTIONS = 3
```

## 📊 Validation Rules

A PDF is accepted as a research paper if:
- ✅ File format is PDF
- ✅ Page count ≥ 4
- ✅ Contains at least 3 standard sections:
  - Abstract, Introduction, Methodology, Results, Conclusion, References, etc.
- ✅ Contains citation patterns:
  - Numbered references [1], [23]
  - Author citations (Smith et al.)
  - DOI patterns (doi:10.xxxx)
  - Year citations (2023), Author (2023)

## 🧪 Testing

Test individual modules:

```bash
# Test validator
python src/validator.py

# Test extractor
python src/extractor.py

# Test embedder
python src/embedder.py

# Test vector store
python src/vectorstore.py
```

## 🔍 Troubleshooting

### "Only research papers are allowed"
- Verify the PDF has at least 4 pages
- Ensure the PDF contains standard research sections
- Check if the document has citations/references

### "Could not find this information"
- The answer may not be in the uploaded papers
- Try rephrasing your question
- Upload more relevant papers

### Slow performance
- First run downloads embedding model (~90MB)
- Reduce `CHUNK_SIZE` or `TOP_K` in config
- Use GPU acceleration for embeddings (install `torch` with CUDA)

## 🚀 Future Enhancements

Potential improvements for version 2.0:

1. **Multi-Paper Analysis**
   - Compare findings across multiple papers
   - Cross-reference citations
   - Identify contradictions or agreements

2. **Advanced Features**
   - Table and figure extraction
   - Equation parsing and rendering
   - ArXiv direct import
   - Export conversations as bibliography

3. **UI Improvements**
   - Citation graph visualization
   - Highlight relevant paper sections
   - Advanced filtering and search
   - Paper similarity analysis

4. **Performance**
   - GPU acceleration
   - Caching for frequent queries
   - Fine-tuned scientific embeddings
   - Incremental indexing

5. **Collaboration**
   - User accounts and saved sessions
   - Shared paper collections
   - Team annotations
   - Export to reference managers

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review ARCHITECTURE.md for system details

## 🙏 Acknowledgments

Built with:
- [Streamlit](https://streamlit.io/) - UI framework
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [PyMuPDF](https://pymupdf.readthedocs.io/) - PDF processing
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [LangChain](https://langchain.com/) - RAG framework

---

**⚠️ Important:** This chatbot answers strictly from uploaded papers. No external knowledge is used. Always verify critical information from the original papers.
