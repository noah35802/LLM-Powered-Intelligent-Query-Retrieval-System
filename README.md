# 🤖 LLM-Powered Intelligent Query–Retrieval System 🤖

## 🎯 Problem Statement

This project aims to build a system that leverages Large Language Models (LLMs) to process natural language queries and retrieve relevant information from large unstructured documents such as policy documents, contracts, and emails.

## ✨ Features

*   **Natural Language Querying:** 🗣️ Ask questions in plain English.
*   **Document Chunking:** 📄 Splits large documents into smaller, manageable chunks for efficient processing.
*   **Semantic Search:** 🧠 Utilizes semantic search to find the most relevant document chunks.
*   **LLM-Powered Answering:** 💡 Uses a Large Language Model to generate answers based on the retrieved information.
*   **PDF Support:** 📑 Can process and extract information from PDF documents.

## 🚀 Real-World Impact

In a world inundated with information, this project is a game-changer! 🌍 Imagine effortlessly querying lengthy legal contracts, dense financial reports, or complex insurance policies just by asking simple questions. This system empowers professionals in various fields to save countless hours of manual work, reduce the risk of overlooking critical information, and make data-driven decisions faster than ever before. From legal professionals to financial analysts, this tool can be the intelligent assistant they've always needed.

## 🎥 Video Demo

[Link to Video Demo]

## ⚙️ How it Works

1.  **Document Ingestion:** The system takes large, unstructured documents (in PDF format) as input.
2.  **Chunking:** The documents are split into smaller chunks to be processed by the LLM.
3.  **Semantic Search:** When a user asks a question, the system performs a semantic search over the document chunks to find the most relevant ones.
4.  **LLM-Powered Response Generation:** The relevant chunks and the user's query are passed to a Large Language Model, which then generates a human-like answer.

## 📂 Project Structure

```
.
├── .env
├── main.py
├── requirements.txt
├── data/
│   └── sample_chunks/
├── documents/
├── prompts/
│   └── decision_prompt.txt
└── utils/
    ├── chunking.py
    ├── file_ops.py
    ├── gemini_client.py
    └── semantic_search.py
```

*   **`.env`**: Environment variables file.
*   **`main.py`**: The main entry point of the application.
*   **`requirements.txt`**: A list of the Python dependencies for the project.
*   **`data/sample_chunks`**: Directory to store the processed chunks of the documents.
*   **`documents`**: Directory to store the original documents.
*   **`prompts/decision_prompt.txt`**: Contains the prompt template for the LLM.
*   **`utils/`**: Contains various utility modules.
    *   **`chunking.py`**: Handles the chunking of documents.
    *   **`file_ops.py`**: Contains file operations.
    *   **`gemini_client.py`**: Handles the interaction with the Gemini API.
    *   **`semantic_search.py`**: Implements the semantic search functionality.

## 🚀 Getting Started

### Prerequisites

*   Python 3.x
*   Pip

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/llm-doc-reasoner-gemini.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd llm-doc-reasoner-gemini
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  Create a `.env` file in the root directory of the project.
2.  Add your Gemini API key to the `.env` file:
    ```
    GEMINI_API_KEY=your_api_key
    ```

### Running the Application

```bash
python main.py
```

## 📝 Usage

1.  Place your PDF documents in the `documents` directory.
2.  Run the application.
3.  Enter your query when prompted.

## 🔮 Future Scope

*   **Support for More Document Formats:**  erweitern to include support for DOCX, TXT, and even HTML files.
*   **Web Interface:** 💻 Develop a user-friendly web interface to make the system more accessible to non-technical users.
*   **User Authentication:** 🔐 Implement user authentication and document-level access control for enhanced security.
*   **Multi-Language Support:** 🌐 Add support for multiple languages to make the system globally applicable.
*   **Domain-Specific Fine-Tuning:** 🎯 Fine-tune the LLM for specific domains (e.g., legal, medical) to improve accuracy and relevance.

## ⚡ Efficiency Improvements

*   **Advanced Chunking:** Implement more sophisticated chunking strategies that are context-aware.
*   **Vector Database:** Utilize a more efficient and scalable vector database for semantic search.
*   **Caching:** Implement caching for LLM responses to common queries to reduce latency and API costs.
*   **Optimized PDF Parsing:** Optimize the PDF parsing process for faster and more accurate text extraction.
*   **Asynchronous Processing:** Implement asynchronous processing for document ingestion and chunking to improve performance.

## 👥 Creators

*   Ayush Raj
*   Aryan Kumar
*   Aryan Bhargava
*   Jatin Kabra
