# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) application built with LangChain for document processing and intelligent question answering.

## 🚀 Features

- **Document Processing**: Support for PDF documents using PyMuPDF and PyPDF
- **LangChain Integration**: Built with LangChain for robust AI workflows
- **RAG Implementation**: Retrieve relevant information and generate accurate responses
- **Modern Python**: Uses Python 3.14 with UV package management

## 📋 Prerequisites

- Python 3.14 or higher
- UV package manager
- Git

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

Run the main application:

```bash
uv run main.py
```

## 📁 Project Structure

```
langchain-rag/
├── main.py              # Main application entry point
├── pyproject.toml       # Project configuration and dependencies
├── requirements.txt     # Package requirements
├── .python-version      # Python version specification
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## 📦 Dependencies

- **langchain**: Core LangChain framework for building AI applications
- **langchain-core**: Essential LangChain components and abstractions
- **langchain-community**: Community-contributed LangChain integrations
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

### Basic RAG Workflow

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load and process documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Query the documents
query = "What is the main topic of this document?"
docs = vectorstore.similarity_search(query)
```

## 🔒 Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI API Key (if using OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here

# Other API keys as needed
ANTHROPIC_API_KEY=your_anthropic_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
```

## 📚 Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [RAG with LangChain Tutorial](https://python.langchain.com/docs/tutorials/rag/)

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
