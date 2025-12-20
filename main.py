"""
Simple FastAPI Backend
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from faq_backend import get_chatbot

app = FastAPI(title="FAQ Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return {"message": "FAQ Chatbot API", "status": "active"}

@app.get("/api/health")
def health():
    return {"status": "healthy", "message": "API is running"}

@app.post("/api/query")
def query(request: QueryRequest):
    chatbot = get_chatbot()
    response = chatbot.get_response(request.query)
    return response

@app.get("/api/faqs")
def get_faqs():
    chatbot = get_chatbot()
    return chatbot.get_all_faqs()

@app.get("/api/categories")
def get_categories():
    chatbot = get_chatbot()
    return chatbot.get_categories()

@app.get("/api/statistics")
def get_stats():
    chatbot = get_chatbot()
    return chatbot.get_statistics()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)