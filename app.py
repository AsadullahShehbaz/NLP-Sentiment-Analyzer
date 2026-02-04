"""
🎬 IMDB Sentiment Analyzer (UI)
Streamlit frontend for sentiment classification
"""

import time
import streamlit as st
import plotly.graph_objects as go

from main import load_model, predict_sentiment

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CACHE MODEL
# ========================================
@st.cache_resource
def get_model():
    return load_model()


def main():
    # Header
    st.title("🎬 IMDB Sentiment Analyzer")
    st.caption("Fine-tuned DistilBERT for movie review sentiment analysis")

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        st.info("""
        **Model:** DistilBERT (fine-tuned)  
        **Dataset:** IMDB Reviews  
        **Accuracy:** 86.25%  
        **Framework:** PyTorch + HuggingFace
        """)

        st.divider()

        st.header("🎯 How to Use")
        st.markdown("""
        1. Enter a movie review  
        2. Click **Analyze Sentiment**  
        3. View prediction & confidence
        """)

    # Load model
    with st.spinner("🔄 Loading model..."):
        tokenizer, model = get_model()

    st.success("✅ Model loaded successfully")

    st.divider()

    # Input
    st.subheader("📝 Enter Movie Review")

    samples = {
        "Positive Example": "This movie was absolutely amazing! The acting was superb.",
        "Negative Example": "Terrible movie. Predictable story and bad acting.",
        "Custom Review": ""
    }

    choice = st.selectbox("Choose sample:", samples.keys())

    user_input = st.text_area(
        "Review Text",
        value=samples[choice],
        height=150,
        placeholder="Write your review here..."
    )

    # Analyze button
    _, col, _ = st.columns([1, 2, 1])
    with col:
        analyze = st.button("🚀 Analyze Sentiment", use_container_width=True)

    if analyze:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review")
            return

        with st.spinner("🤖 Analyzing..."):
            time.sleep(0.4)
            result = predict_sentiment(user_input, tokenizer, model)

        st.divider()
        st.subheader("📊 Results")

        st.metric("Sentiment", result["sentiment"])
        st.metric("Confidence", f"{result['confidence']*100:.2f}%")

        # Chart
        fig = go.Figure([
            go.Bar(
                x=["Negative", "Positive"],
                y=[
                    result["negative_prob"] * 100,
                    result["positive_prob"] * 100
                ],
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
