"""
NLP Model Logic - Handles model loading and inference
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "asadullahshehbaz/my_text_classifier_model"


def load_model():
    """
    Load fine-tuned DistilBERT model and tokenizer.
    
    Returns:
        tuple: (tokenizer, model)
    """
    try:
        print(f"Loading tokenizer from {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print(f"Loading model from {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()
        
        print("Model and tokenizer loaded successfully")
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise


def predict_sentiment(text: str, tokenizer, model):
    """
    Predict sentiment for a given text.
    
    Args:
        text (str): Input review text
        tokenizer: Tokenizer instance
        model: Model instance
    
    Returns:
        dict: {
            'sentiment': str,
            'confidence': float,
            'positive_prob': float,
            'negative_prob': float
        }
    """
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    negative_prob = probs[0][0].item()
    positive_prob = probs[0][1].item()
    
    sentiment = "Positive 😊" if positive_prob > negative_prob else "Negative 😞"
    confidence = max(positive_prob, negative_prob)
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_prob": positive_prob,
        "negative_prob": negative_prob,
    }