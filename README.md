# 🩺 AI Medical Assistant (RAG Pipeline)

An advanced, production-ready Retrieval-Augmented Generation (RAG) application designed to parse complex medical documents and provide rapid, highly contextual answers based entirely on user-uploaded data. 

This project utilizes a decoupled **Client-Server architecture**, combining a fast, lightweight frontend UI with an asynchronous backend processing engine.

---

## 🚀 Key Features

* **Dynamic PDF Processing:** Seamlessly upload and parse complex multi-page PDF documents.
* **Smart Text Chunking:** Utilizes semantic text splitting to divide text into contextual chunks while maintaining document metadata.
* **High-Dimensional Vector Search:** Generates state-of-the-art text embeddings via Google's `gemini-embedding-001` and indexes them into a high-performance `Pinecone` serverless vector database.
* **Ultra-Fast Generation Engine:** Leverages the `Groq` inference engine powered by `Llama-3.3-70b-versatile` to formulate precise, human-like answers with an extensive 128k context window.
* **Interactive UI:** Built entirely on Streamlit to offer an intuitive, chat-like experience.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **Backend Framework** | FastAPI + Uvicorn |
| **RAG Orchestration** | LangChain |
| **Embedding Model** | Google Gemini (`models/gemini-embedding-001`) |
| **Vector Database** | Pinecone (Serverless Vector DB) |
| **LLM Inference** | Groq Cloud (`llama-3.3-70b-versatile`) |

---

## 📁 Architecture Overview

```text
├── client/
│   ├── app.py              # Main Streamlit application
│   ├── components/         # UI structural elements (Chat UI, loaders)
│   └── utils/              # API client network configurations
├── server/
│   ├── main.py             # FastAPI server entry point
│   ├── routes/             # API Endpoints (/upload_pdfs, /ask)
│   └── modules/            # Vectorstore loaders, PDF handlers, and LLM utilities
├── requirements.txt        # Shared / Deployment dependencies
└── .gitignore              # Local configuration exclusion mappings

## Getting Started
1. Clone the Repository
git clone [https://github.com/aniketbhaumik/MedicalAssistant.git](https://github.com/aniketbhaumik/MedicalAssistant.git)
cd MedicalAssistant

2. Environment Setup
Create a .env file in your server/ directory and add your respective API keys:
GOOGLE_API_KEY="your-google-api-key-here"
PINECONE_API_KEY="your-pinecone-api-key-here"
GROQ_API_KEY="your-groq-api-key-here"

3. Running Locally
Because this application uses a decoupled Client-Server architecture, you must run the backend and frontend simultaneously in two separate terminal windows:

🔸 Terminal 1: Run the Backend Server (FastAPI)
cd server
uvicorn main:app --reload

🔸 Terminal 2: Run the Frontend UI (Streamlit)
cd client
streamlit run app.py
