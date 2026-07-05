# 📄 PDF Chatbot with Streamlit and Google Gemini (RAG)

This project implements a conversational AI chatbot that can answer questions based on the content of uploaded PDF documents. It leverages Streamlit for the user interface, LangChain for orchestrating the RAG (Retrieval Augmented Generation) pipeline, and Google's Gemini models for embeddings and language generation.

## ✨ Features

- PDF Upload: Easily upload any PDF document.
- Text Extraction: Automatically extracts text content from the uploaded PDF.
- Text Chunking: Divides the extracted text into manageable chunks for efficient processing.
- Vector Embeddings: Generates numerical representations (embeddings) of text chunks using Google's gemini-embedding-2-preview model.
- Vector Store (FAISS): Stores and indexes the embeddings for fast similarity search.
- Retrieval Augmented Generation (RAG): When a user asks a question, the system retrieves relevant text chunks from the PDF and uses them as context for the Gemini LLM to generate an accurate answer.
- Conversational AI: Provides a natural language interface for querying your documents.
- Streamlit UI: A simple and intuitive web interface for interaction.

---

## 🚀 Getting Started

Follow these steps to set up and run the PDF Chatbot locally.

### 📋 Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.8+
- pip (Python package installer)
- Google API Key: You'll need an API key from Google AI Studio (formerly Google Cloud Console for Generative AI). Get yours here: Google AI Studio

### 📦 Installation

Clone the repository (or download the files):

```bash
git clone https://github.com/your-username/pdf-chatbot-rag.git
cd pdf-chatbot-rag
```

(Replace your-username with your actual GitHub username if you've forked it, or adjust the path if you downloaded directly.)

Create a virtual environment (HIGHLY RECOMMENDED): A virtual environment (venv) isolates your project's dependencies from other Python projects and your system's Python installation, preventing conflicts.

```bash
python -m venv venv
```

Activate the virtual environment: You must activate the virtual environment in each new terminal session before running the application or installing packages.

Windows:

```bash
.\venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

You'll typically see (venv) or a similar indicator in your terminal prompt when the virtual environment is active.

Install the required Python packages: With your virtual environment active, install the necessary libraries.

```bash
pip install -r requirements.txt
```

(If you don't have a requirements.txt yet, create one with the following content and then run the command:)

```text
streamlit
python-dotenv
langchain
langchain-google-genai
pdfplumber
faiss-cpu
```

### 🔑 Set Up Your Google API Key

The application requires a Google API Key to access the Gemini models. It's best practice to store this key as an environment variable.

Create a .env file: In the root directory of your project (the same directory as rag.py), create a file named .env.

Add your API key to .env: Open the .env file and add the following line, replacing YOUR_ACTUAL_GOOGLE_API_KEY with the key you obtained from Google AI Studio:

```env
GOOGLE_API_KEY=YOUR_ACTUAL_GOOGLE_API_KEY
```

Important: Do not commit your .env file to version control (e.g., Git). Add .env to your .gitignore file.

### ▶️ Running the Application

Ensure your virtual environment is active. (If you closed your terminal, reactivate it using the command from step 3 in "Installation").

Run the Streamlit application:

```bash
streamlit run rag.py
```

This command will open the chatbot in your default web browser (usually at http://localhost:8501).

### 💡 How to Use

- Upload PDF: On the left sidebar, click the "Browse files" button to upload a PDF document.
- Processing: The application will automatically extract text, chunk it, generate embeddings, and store them in a vector database. You'll see a "PDF processed and embeddings generated!" message upon success.
- Ask Questions: Once the PDF is processed, a text input field labeled "Type your question here" will appear. Enter your question related to the PDF content.
- Get Answers: The chatbot will retrieve relevant information from the PDF and generate an answer using the Gemini LLM, which will be displayed below your question.

### ⚙️ Project Structure

```text
.
├── rag.py                  # Main Streamlit application script
├── .env                    # Stores your Google API Key (ignored by Git)
├── .gitignore              # Specifies files/directories to ignore (e.g., .env, venv/)
├── requirements.txt        # Lists all Python dependencies
└── venv/                   # Python virtual environment directory (created by `python -m venv venv`)
```

### 🧠 Core Concepts

This project demonstrates a Retrieval Augmented Generation (RAG) system, which combines the power of large language models (LLMs) with external knowledge sources (your PDFs) to provide more accurate and context-aware answers.

Here's a breakdown of the RAG pipeline implemented:

- Document Loading & Text Extraction (pdfplumber):

The user uploads a PDF.
pdfplumber is used to open the PDF and extract all textual content page by page.

- Text Splitting (RecursiveCharacterTextSplitter from langchain_text_splitters):

Large documents are broken down into smaller, overlapping "chunks" of text. This is crucial because LLMs have token limits, and smaller chunks are easier to embed and retrieve.
RecursiveCharacterTextSplitter intelligently splits text based on a list of separators, ensuring chunks maintain semantic coherence where possible.

- Embedding Generation (GoogleGenerativeAIEmbeddings):

Each text chunk is converted into a high-dimensional numerical vector (an "embedding").
GoogleGenerativeAIEmbeddings uses Google's gemini-embedding-2-preview model to create these embeddings. Embeddings capture the semantic meaning of the text, so similar texts have similar vector representations.

- Vector Store (FAISS):

The generated embeddings are stored in a vector database. FAISS (Facebook AI Similarity Search) is an efficient library for similarity search on large sets of vectors.
This allows for rapid searching of relevant text chunks based on their semantic similarity to a query.

- Retrieval (vector_store.as_retriever()):

When a user asks a question, that question is also converted into an embedding.
The vector store is queried to find the k most semantically similar text chunks from the original PDF. This is the "Retrieval" part of RAG.
search_type="mmr" (Maximum Marginal Relevance) is used to select diverse yet relevant documents, avoiding redundancy.

- Language Model (LLM) (ChatGoogleGenerativeAI):

ChatGoogleGenerativeAI interfaces with Google's gemini-3.5-flash model.
The retrieved text chunks (context) and the user's original question are combined into a single prompt.

- Prompt Engineering (ChatPromptTemplate):

A ChatPromptTemplate is used to structure the input to the LLM. It includes:
A "system" message providing instructions to the LLM (e.g., "You are a helpful assistant...", "Only use information from the provided context...").
The retrieved context from the PDF.
The human user's question.
This ensures the LLM focuses on the provided context and adheres to specific answering guidelines.

- Output Parsing (StrOutputParser):

The LLM's raw output (which might be a complex object) is parsed into a simple string format for display.

- LangChain Chain (RunnablePassthrough, | operator):

LangChain's Runnable interface and the | (pipe) operator are used to create a sequential chain of operations:
{"context": retriever | format_docs, "question": RunnablePassthrough()}: This prepares the input for the prompt. The retriever gets documents, format_docs formats them, and RunnablePassthrough() passes the original question through.
| prompt: The prepared input is fed into the ChatPromptTemplate.
| llm: The templated prompt goes to the Gemini LLM.
| StrOutputParser(): The LLM's response is parsed.
This chain orchestrates the entire RAG process from question to answer.

### ⚠️ Error Handling

The application includes basic error handling for the embedding generation step. If there's an issue with your Google API Key or the embedding model, an error message will be displayed in the Streamlit UI, guiding you to check your key and model name.

### 🤝 Contributing

Feel free to fork this repository, open issues, or submit pull requests to improve the project.

### 📄 License

This project is open-source and available under the MIT License.
