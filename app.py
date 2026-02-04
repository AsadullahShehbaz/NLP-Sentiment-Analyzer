import time
import streamlit as st
import plotly.graph_objects as go
import sys
from logg import logger 

from main import load_model, predict_sentiment


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Production-Ready NLP Sentiment Analysis System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CACHE MODEL
# -------------------------------------------------
@st.cache_resource
def get_model():
    logger.info("Loading model & tokenizer …")
    tokenizer, model = load_model()
    logger.info("Model loaded successfully")
    return tokenizer, model

def main():
    st.title("🎬 Production-Ready NLP Sentiment Analyzer")
    st.caption("Fine‑tuned DistilBERT for movie review sentiment analysis")

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("📊 Model Info")
        st.info(
            """
            **Model:** DistilBERT (fine‑tuned)  
            **Dataset:** IMDB Reviews  
            **Accuracy:** 86.25%  
            **Framework:** PyTorch + HuggingFace
            """
        )
        st.divider()
        st.header("🎯 How to Use")
        st.markdown(
            """
            1. Enter a movie review  
            2. Click **Analyze Sentiment**  
            3. View prediction & confidence
            """
        )
        logger.debug("Sidebar rendered")

    # ---------- Load model ----------
    with st.spinner("🔄 Loading model…"):
        tokenizer, model = get_model()
    st.success("✅ Model loaded successfully")

    st.divider()

    # ---------- Input ----------
    st.subheader("📝 Enter Movie Review")
    samples = {
        "Positive Example": "This movie was absolutely amazing! The acting was impressive .",
        "Negative Example": "Terrible movie. Predictable story and bad acting.",
        "Custom Review": ""
    }
    choice = st.selectbox("Choose sample:", samples.keys())
    logger.debug(f"Sample choice: {choice}")

    user_input = st.text_area(
        "Review Text",
        value=samples[choice],
        height=150,
        placeholder="Write your review here..."
    )

    # ---------- Analyze ----------
    _, col, _ = st.columns([1, 2, 1])
    with col:
        analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

    if analyze:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review")
            logger.warning("Analyze clicked with empty review")
            return

        logger.info(f"User review (truncated): {user_input[:75]!r}…")
        try:
            with st.spinner("🤖 Analyzing…"):
                time.sleep(0.4)
                result = predict_sentiment(user_input, tokenizer, model)
        except Exception as exc:
            logger.exception("Prediction failed")
            st.error("❌ An error occurred while analyzing the review.")
            return

        logger.info(
            f"Result – Sentiment: {result['sentiment']}, "
            f"Confidence: {result['confidence']:.2%}"
        )
        logger.debug(
            f"Probabilities – Neg: {result['negative_prob']:.4f}, "
            f"Pos: {result['positive_prob']:.4f}"
        )

        # ---------- Display results ----------
        st.divider()
        st.subheader("📊 Results")
        st.metric("Sentiment", result["sentiment"])
        st.metric("Confidence", f"{result['confidence']*100:.2f}%")

        fig = go.Figure([
            go.Bar(
                x=["Negative", "Positive"],
                y=[result["negative_prob"]*100, result["positive_prob"]*100],
                text=[
                    f"{result['negative_prob']*100:.2f}%",
                    f"{result['positive_prob']*100:.2f}%"
                ],
                textposition="auto"
            )
        ])
        fig.update_layout(
            title="Sentiment Probability Distribution",
            yaxis_range=[0, 100],
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Built with ❤️ using Streamlit & HuggingFace")

if __name__ == "__main__":
    main()