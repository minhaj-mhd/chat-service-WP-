# 💬 Chat Service - FastAPI Web Content Q&A

The **Chat Service** is a FastAPI microservice that enables users to ask questions about previously parsed webpage content. It uses **HuggingFace Transformers** for question answering and **Sentence Transformers** for embedding (prepared for future enhancements). The service retrieves content from an **SQLite** database, where the `web_processor_service` has stored the parsed webpage data.

---

## 🚀 Features

- Exposes a `/ask/` API endpoint to answer questions about stored web content.
- Uses:
  - `distilbert-base-cased-distilled-squad` for Question Answering
  - `paraphrase-MiniLM-L6-v2` for future embedding-related tasks
- Connects to an SQLite database (`content.db`) shared with the web processor service
- Implements proper error handling and CORS support

---

## ⚙️ Tech Stack

- FastAPI
- SQLite3
- HuggingFace Transformers (`pipeline`)
- Sentence Transformers
- Pydantic
- CORS Middleware (for cross-origin API access)

---

