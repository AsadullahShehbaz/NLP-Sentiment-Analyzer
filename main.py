"""
🧠 Core Logic for IMDB Sentiment Classifier
Handles model loading and inference only
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from logg import logger

# -------------------------------------------------
# Model constants
# -------------------------------------------------
MODEL_NAME = "asadullahshehbaz/my_text_classifier_model"


def load_model():
    """
    Load fine‑tuned DistilBERT model and tokenizer.
    Returns:
        tokenizer, model
    """
    try:
        logger.info("Loading tokenizer from %s", MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        logger.info("Loading model from %s", MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()

        logger.info("Model and tokenizer loaded successfully.")
        return tokenizer, model
    except Exception as e:
        logger.exception("Failed to load model or tokenizer: %s", e)
        raise


def predict_sentiment(text: str, tokenizer, model):
    """
    Predict sentiment for a given text.

    Args:
        text (str): Input sentence / review.
        tokenizer: Tokenizer instance.
        model: Model instance.

    Returns:
        dict with sentiment, confidence, positive_prob, negative_prob
    """
    logger.debug("Received text for prediction: %s", text)

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    logger.debug("Tokenized inputs: %s", inputs)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    negative_prob = probs[0][0].item()
    positive_prob = probs[0][1].item()

    sentiment = "Positive 😊" if positive_prob > negative_prob else "Negative 😞"
    confidence = max(positive_prob, negative_prob)

    logger.info(
        "Predicted Sentiment: %s | Confidence: %.4f | Pos: %.4f | Neg: %.4f",
        sentiment,
        confidence,
        positive_prob,
        negative_prob,
    )

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_prob": positive_prob,
        "negative_prob": negative_prob,
    }