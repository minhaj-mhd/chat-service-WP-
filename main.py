from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sqlite3
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#load models

embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

conn = sqlite3.connect("../web_processor/content.db")

class ChatRequest(BaseModel):
    content_id: str
    question:str

@app.post("/ask/")
async def ask_question(request:ChatRequest):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT text FROM processed_content WHERE id = ?",(request.content_id,))
        result = cursor.fetchone()

        if not result:
            raise HTTPException(status_code = 404, detail = "Content not found")
        
        context = result[0][:1000]
        answer = qa_model(question=request.question, context = context)

        return {
            "answer": answer["answer"],
            "confidence": float(answer["score"]),
            "context_used": len(context)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))