# LangChain RAG Project

A Retrieval-Augmented Generation (RAG) application built with LangChain for document processing and intelligent question answering.

## 🚀 Features

- **Document Processing**: Support for PDF and text documents using PyMuPDF and PyPDF
- **LangChain Integration**: Built with LangChain for robust AI workflows
- **RAG Implementation**: Retrieve relevant information and generate accurate responses
- **Jupyter Notebook Support**: Interactive development with document.ipynb
- **Sample Data**: Pre-configured text and PDF files for testing
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

### Running the Main Application

```bash
uv run main.py
```

### Using Jupyter Notebook

1. **Start Jupyter notebook:**

   ```bash
   uv run jupyter notebook
   ```

2. **Open the example notebook:**
   Navigate to `notebook/document.ipynb` for interactive document processing examples

3. **Alternative - Use VS Code:**
   Open `notebook/document.ipynb` directly in VS Code with the Python extension

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
    └── document.ipynb       # Interactive document processing examples
```

## 📦 Dependencies

- **ipykernel**: Jupyter kernel for running Python notebooks
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

Check out `notebook/document.ipynb` for comprehensive examples including:

- **Document Loading**: Loading text and PDF files
- **Text Processing**: Using TextLoader and DirectoryLoader
- **PDF Processing**: Working with PyPDFLoader and PyMuPDFLoader
- **Document Structure**: Understanding LangChain Document objects

### Basic RAG Workflow (Python Script)

```python
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load documents from directory
dir_loader = DirectoryLoader(
    "data/text_files",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'}
)
documents = dir_loader.load()

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
query = "What is Python programming?"
docs = vectorstore.similarity_search(query)
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
