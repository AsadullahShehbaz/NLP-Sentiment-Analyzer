"""
🧠 Core Logic for IMDB Sentiment Classifier
Handles model loading and inference only
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "asadullahshehbaz/my_text_classifier_model"


def load_model():
    """
    Load fine-tuned DistilBERT model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_sentiment(text: str, tokenizer, model):
    """
    Predict sentiment for given text
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

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
        "negative_prob": negative_prob
    }
