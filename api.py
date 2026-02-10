"""
FastAPI Backend - REST API for Sentiment Analysis
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model import load_model, predict_sentiment

# Global model storage
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    print("🚀 Starting up - Loading model...")
    tokenizer, model = load_model()
    ml_models["tokenizer"] = tokenizer
    ml_models["model"] = model
    print("✅ Model loaded successfully")
    yield
    print("🛑 Shutting down - Cleaning up...")
    ml_models.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready NLP sentiment analyzer using DistilBERT",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ReviewRequest(BaseModel):
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "This movie was absolutely amazing! Great acting and story."
            }
        }


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    positive_prob: float
    negative_prob: float


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Sentiment Analysis API",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": bool(ml_models)
    }


@app.post("/predict", response_model=SentimentResponse)
async def analyze_sentiment(request: ReviewRequest):
    """
    Analyze sentiment of review text
    
    Args:
        request: ReviewRequest with text field
    
    Returns:
        SentimentResponse with sentiment, confidence, and probabilities
    """
    # Validate input
    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Review text cannot be empty"
        )
    
    # Check if model is loaded
    if not ml_models:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )
    
    # Predict sentiment
    try:
        result = predict_sentiment(
            request.text,
            ml_models["tokenizer"],
            ml_models["model"]
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)