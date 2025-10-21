# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) application built with LangChain for document processing and intelligent question answering.

## 🚀 Features

- **Advanced Document Processing**: Support for PDF and text documents using PyMuPDF and PyPDF
- **Smart Text Splitting**: Intelligent document chunking with RecursiveCharacterTextSplitter
- **ChromaDB Vector Store**: Persistent vector database with sentence transformers embeddings
- **Complete RAG Pipeline**: End-to-end data ingestion to query response workflow
- **Multiple LLM Support**: Integration with Groq's Llama models for text generation
- **Advanced Retrieval**: Similarity search with configurable thresholds and ranking
- **Enhanced RAG Features**: Citations, confidence scoring, query history, and streaming
- **Production-Ready Classes**: Modular EmbeddingManager, VectorStore, and RAGRetriever components
- **Jupyter Notebook Support**: Interactive development with comprehensive examples
- **Sample Data**: Pre-configured text and PDF files for testing
- **Modern Python**: Uses Python 3.13 with UV package management

## 📋 Prerequisites

- Python 3.13 or higher
- UV package manager
- Git
- Groq API Key (for LLM functionality)

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd langchain-rag
   ```

2. **Create and activate virtual environment:**

   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   uv add -r requirements.txt
   ```

## 🏃‍♂️ Quick Start

### Running the Main Application

```bash
uv run main.py
```

### Using Jupyter Notebook

1. **Start Jupyter notebook:**

   ```bash
   uv run jupyter notebook
   ```

2. **Open the example notebooks:**

   - `notebook/document.ipynb` - Basic document processing examples
   - `notebook/pdf_loader.ipynb` - Complete RAG pipeline with PDF processing

3. **Alternative - Use VS Code:**
   Open notebook files directly in VS Code with the Python extension for the best development experience

## 📁 Project Structure

```
langchain-rag/
├── main.py                    # Main application entry point
├── pyproject.toml            # Project configuration and dependencies
├── requirements.txt          # Package requirements
├── .python-version           # Python version specification
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── data/                    # Data directory
│   ├── text_files/          # Sample text documents
│   │   ├── python_intro.txt
│   │   └── machine_learning.txt
│   └── pdf_files/           # PDF documents for processing
└── notebook/                # Jupyter notebooks
    ├── document.ipynb       # Basic document processing examples
    └── pdf_loader.ipynb     # Complete RAG pipeline with ChromaDB and Groq LLM
```

## 📦 Dependencies

- **chromadb**: Vector database for persistent embeddings storage
- **sentence-transformers**: State-of-the-art sentence embeddings
- **langchain-groq**: Groq LLM integration for fast inference
- **scikit-learn**: Machine learning utilities for similarity calculations
- **numpy**: Numerical computing for embeddings processing
- **ipykernel**: Jupyter kernel for running Python notebooks
- **langchain**: Core LangChain framework for building AI applications
- **langchain-core**: Essential LangChain components and abstractions
- **langchain-community**: Community-contributed LangChain integrations
- **langchain-text-splitters**: Advanced text splitting utilities for document chunking
- **pydantic-settings**: Configuration management with Pydantic V2 compatibility
- **pypdf**: Pure Python PDF library for reading PDF files
- **pymupdf**: Python bindings for MuPDF (fast PDF processing)

## 🔧 Development

### Adding New Dependencies

```bash
# Add a new package
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add from requirements file
uv add -r requirements.txt
```

### Running Tests

```bash
# Add pytest for testing
uv add --dev pytest

# Run tests
uv run pytest
```

### Working with Jupyter Notebooks

```bash
# Install additional Jupyter tools
uv add --dev notebook jupyterlab

# Start Jupyter Lab
uv run jupyter lab

# Start classic Jupyter Notebook
uv run jupyter notebook
```

### Code Formatting

```bash
# Add development tools
uv add --dev black isort flake8

# Format code
uv run black .
uv run isort .

# Lint code
uv run flake8 .
```

## 🤖 Usage Examples

### Interactive Notebook Examples

**1. Basic Document Processing (`notebook/document.ipynb`):**

- Document loading with TextLoader and DirectoryLoader
- Working with LangChain Document objects
- Basic PDF processing examples

**2. Complete RAG Pipeline (`notebook/pdf_loader.ipynb`):**

- **Production-Ready Classes**: EmbeddingManager, VectorStore, and RAGRetriever
- **ChromaDB Integration**: Persistent vector storage with sentence transformers
- **Batch PDF Processing**: Process multiple PDFs with error handling
- **Advanced Text Splitting**: Optimized chunking with metadata preservation
- **Groq LLM Integration**: Fast inference with Llama-3.1 models
- **Enhanced RAG Features**: Citations, confidence scoring, streaming responses
- **Query History**: Track and analyze query patterns
- **Configurable Retrieval**: Similarity thresholds and result ranking
- **Error Handling & Debugging**: Comprehensive error handling and diagnostic tools

### Key Features from Notebooks

```python
# Production-Ready Classes (from pdf_loader.ipynb)

# 1. Embedding Manager
class EmbeddingManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

# 2. ChromaDB Vector Store
class VectorStore:
    def __init__(self, collection_name="pdf_documents"):
        self.client = chromadb.PersistentClient(path="../data/vector_store")
        self.collection = self.client.get_or_create_collection(collection_name)

    def add_documents(self, documents, embeddings):
        # Add documents with metadata to ChromaDB
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadata)

# 3. RAG Retriever with Similarity Search
class RAGRetriever:
    def retrieve(self, query, top_k=5, score_threshold=0.0):
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()], n_results=top_k
        )
        return processed_results

# 4. Enhanced RAG Pipeline
def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2):
    """RAG with citations, confidence scoring, and source tracking"""
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    context = "\n\n".join([doc['content'] for doc in results])

    # Generate answer with citations
    response = llm.invoke([f"Context: {context}\nQuestion: {query}\nAnswer:"])

    return {
        'answer': response.content,
        'sources': [{'source': doc['metadata']['source_file'],
                    'score': doc['similarity_score']} for doc in results],
        'confidence': max([doc['similarity_score'] for doc in results])
    }

# 5. Advanced Pipeline with Streaming and History
class AdvancedRAGPipeline:
    def query(self, question, stream=False, summarize=False):
        # Retrieval, generation, citations, history tracking
        return {
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }
```

### Basic RAG Workflow (Python Script)

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# 1. Load and process documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Initialize LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)

# 5. Query the RAG system
def rag_query(question):
    # Similarity search
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate response
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    response = llm.invoke([prompt])
    return response.content

# Example usage
answer = rag_query("What is machine learning?")
print(answer)
```

### Sample Data

The project includes sample data files:

- **Text files** (`data/text_files/`):

  - `python_intro.txt`: Introduction to Python programming
  - `machine_learning.txt`: Machine learning basics

- **PDF files** (`data/pdf_files/`): Add your own PDF documents for processing

## 🔒 Environment Variables

Create a `.env` file in the project root:

```env
# Groq API Key (required for LLM functionality)
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API Key (optional, for OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here

# Other API keys as needed
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

### Getting API Keys

1. **Groq API Key**:

   - Sign up at [console.groq.com](https://console.groq.com)
   - Navigate to API Keys section
   - Create a new API key

2. **Supported Groq Models**:
   - `llama-3.1-70b-versatile` (recommended for quality)
   - `llama-3.1-8b-instant` (recommended for speed)
   - `mixtral-8x7b-32768`
   - `gemma2-9b-it`

## 📚 Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API Documentation](https://console.groq.com/docs)
- [UV Documentation](https://docs.astral.sh/uv/)
- [RAG with LangChain Tutorial](https://python.langchain.com/docs/tutorials/rag/)

## 🎯 Advanced Features

The notebook implements several advanced RAG features:

### 🏗️ **Production Architecture**

- **Modular Design**: Separate classes for embeddings, vector store, and retrieval
- **Error Handling**: Comprehensive error handling and debugging tools
- **Persistence**: ChromaDB for persistent vector storage
- **Scalability**: Batch processing and efficient similarity search

### 🔍 **Enhanced Retrieval**

- **Configurable Similarity**: Adjustable similarity thresholds
- **Source Tracking**: Metadata preservation and source attribution
- **Confidence Scoring**: Retrieval confidence metrics
- **Ranking**: Smart result ranking and filtering

### 🤖 **Advanced Generation**

- **Citations**: Automatic source citation in responses
- **Streaming**: Real-time response streaming simulation
- **History**: Query and response history tracking
- **Summarization**: Optional answer summarization

### 📊 **Analytics & Debugging**

- **Performance Metrics**: Embedding dimensions, similarity scores
- **Debug Tools**: API key validation, context inspection
- **Error Recovery**: Graceful error handling and fallbacks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

If you have any questions or run into issues, please:

- Check the [documentation](https://docs.langchain.com/)
- Open an issue on GitHub
- Contact the maintainers

---

**Happy coding! 🎉**
