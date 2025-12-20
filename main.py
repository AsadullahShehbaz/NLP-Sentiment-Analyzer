"""
FASTAPI BACKEND ROUTER
======================
REST API endpoints for FAQ Chatbot

File: main.py

Installation:
    pip install fastapi uvicorn pydantic

Run Command:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API Documentation:
    http://localhost:8000/docs

Author: AI Engineering Intern
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

# Import your backend
from faq_backend import get_chatbot, FAQ_DATABASE

# ==================== FASTAPI APP SETUP ====================
app = FastAPI(
    title="FAQ Chatbot API",
    description="NLP-based FAQ Chatbot using NLTK and Machine Learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PYDANTIC MODELS ====================

class QueryRequest(BaseModel):
    """Request model for user query"""
    query: str = Field(..., min_length=1, max_length=500, description="User's question")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What are your business hours?"
            }
        }


class QueryResponse(BaseModel):
    """Response model for chatbot answer"""
    success: bool = Field(..., description="Whether query was successful")
    answer: Optional[str] = Field(None, description="Chatbot's answer")
    message: Optional[str] = Field(None, description="Error or info message")
    matched_question: Optional[str] = Field(None, description="Matched FAQ question")
    category: Optional[str] = Field(None, description="FAQ category")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    cosine_score: Optional[float] = Field(None, description="Cosine similarity score")
    keyword_score: Optional[float] = Field(None, description="Keyword match score")
    faq_id: Optional[int] = Field(None, description="Matched FAQ ID")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "answer": "Our business hours are Monday to Friday...",
                "matched_question": "What are your business hours?",
                "category": "General",
                "confidence": 0.85,
                "cosine_score": 0.82,
                "keyword_score": 0.75,
                "faq_id": 1,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class FAQItem(BaseModel):
    """Model for single FAQ item"""
    id: int
    question: str
    answer: str
    keywords: List[str]
    category: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "question": "What are your business hours?",
                "answer": "Our business hours are...",
                "keywords": ["hours", "time", "open"],
                "category": "General"
            }
        }


class StatisticsResponse(BaseModel):
    """Model for chatbot statistics"""
    total_queries: int
    successful_matches: int
    average_confidence: float
    categories_used: List[str]
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


# ==================== API ENDPOINTS ====================

@app.get("/", response_model=Dict)
async def root():
    """
    Root endpoint - API information
    """
    return {
        "message": "FAQ Chatbot API",
        "version": "1.0.0",
        "status": "active",
        "documentation": "/docs",
        "endpoints": {
            "query": "/api/query",
            "faqs": "/api/faqs",
            "categories": "/api/categories",
            "statistics": "/api/statistics",
            "health": "/api/health"
        }
    }


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "message": "FAQ Chatbot API is running"
    }


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process user query and return chatbot response
    
    **Parameters:**
    - **query**: User's question (1-500 characters)
    
    **Returns:**
    - Chatbot response with answer and confidence scores
    """
    try:
        # Get chatbot instance
        chatbot = get_chatbot()
        
        # Process query
        response = chatbot.get_response(request.query)
        
        # Return response
        return QueryResponse(**response)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/api/faqs", response_model=List[FAQItem])
async def get_all_faqs():
    """
    Get all FAQs in the database
    
    **Returns:**
    - List of all FAQ items with questions, answers, and metadata
    """
    try:
        chatbot = get_chatbot()
        faqs = chatbot.get_all_faqs()
        return [FAQItem(**faq) for faq in faqs]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving FAQs: {str(e)}"
        )


@app.get("/api/faqs/{faq_id}", response_model=FAQItem)
async def get_faq_by_id(faq_id: int):
    """
    Get specific FAQ by ID
    
    **Parameters:**
    - **faq_id**: FAQ identifier
    
    **Returns:**
    - Single FAQ item
    """
    try:
        chatbot = get_chatbot()
        faqs = chatbot.get_all_faqs()
        
        # Find FAQ by ID
        faq = next((f for f in faqs if f['id'] == faq_id), None)
        
        if faq is None:
            raise HTTPException(
                status_code=404,
                detail=f"FAQ with ID {faq_id} not found"
            )
        
        return FAQItem(**faq)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving FAQ: {str(e)}"
        )


@app.get("/api/categories", response_model=List[str])
async def get_categories():
    """
    Get all unique FAQ categories
    
    **Returns:**
    - List of category names
    """
    try:
        chatbot = get_chatbot()
        categories = chatbot.get_categories()
        return sorted(categories)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving categories: {str(e)}"
        )


@app.get("/api/faqs/category/{category_name}", response_model=List[FAQItem])
async def get_faqs_by_category(category_name: str):
    """
    Get FAQs filtered by category
    
    **Parameters:**
    - **category_name**: Category to filter by
    
    **Returns:**
    - List of FAQs in the specified category
    """
    try:
        chatbot = get_chatbot()
        faqs = chatbot.get_all_faqs()
        
        # Filter by category (case-insensitive)
        filtered_faqs = [
            faq for faq in faqs 
            if faq['category'].lower() == category_name.lower()
        ]
        
        if not filtered_faqs:
            raise HTTPException(
                status_code=404,
                detail=f"No FAQs found in category '{category_name}'"
            )
        
        return [FAQItem(**faq) for faq in filtered_faqs]
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving FAQs by category: {str(e)}"
        )


@app.get("/api/statistics", response_model=StatisticsResponse)
async def get_statistics():
    """
    Get chatbot usage statistics
    
    **Returns:**
    - Total queries, successful matches, average confidence, categories used
    """
    try:
        chatbot = get_chatbot()
        stats = chatbot.get_statistics()
        return StatisticsResponse(**stats)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving statistics: {str(e)}"
        )


@app.get("/api/history", response_model=List[Dict])
async def get_conversation_history():
    """
    Get conversation history
    
    **Returns:**
    - List of all queries and responses
    """
    try:
        chatbot = get_chatbot()
        history = chatbot.get_conversation_history()
        
        # Convert to serializable format
        serialized_history = []
        for item in history:
            serialized_history.append({
                'query': item['query'],
                'response': item['response'],
                'timestamp': item['timestamp'].isoformat()
            })
        
        return serialized_history
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving history: {str(e)}"
        )


@app.delete("/api/history", response_model=Dict)
async def clear_history():
    """
    Clear conversation history
    
    **Returns:**
    - Confirmation message
    """
    try:
        chatbot = get_chatbot()
        chatbot.conversation_history.clear()
        
        return {
            "success": True,
            "message": "Conversation history cleared",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing history: {str(e)}"
        )


# ==================== ERROR HANDLERS ====================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not Found",
        "message": "The requested resource was not found",
        "status_code": 404
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    return {
        "error": "Internal Server Error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }


# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("=" * 60)
    print("FAQ Chatbot API Starting...")
    print("=" * 60)
    print("API Documentation: http://localhost:8000/docs")
    print("API Base URL: http://localhost:8000")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("FAQ Chatbot API Shutting Down...")


# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )