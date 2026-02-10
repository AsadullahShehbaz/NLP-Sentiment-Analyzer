"""
Streamlit Frontend - Web UI for Sentiment Analysis
Connects to FastAPI backend via HTTP requests
"""
import streamlit as st
import plotly.graph_objects as go
import requests


# Page config
st.set_page_config(
    page_title="Sentiment Analysis System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# API Configuration
API_URL = "http://localhost:8000"


def check_api_health():
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def predict_sentiment(text: str):
    """Call FastAPI backend to predict sentiment"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise Exception("❌ Cannot connect to API. Make sure backend is running at " + API_URL)
    except Exception as e:
        raise Exception(f"❌ Error: {str(e)}")


def main():
    st.title("🎬 NLP Sentiment Analyzer")
    st.caption("Fine-tuned DistilBERT for movie review sentiment analysis")

    # Sidebar
    with st.sidebar:
        st.header("📊 Model Info")
        st.info(
            """
            **Model:** DistilBERT (fine-tuned)  
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
        
        st.divider()
        # API health check
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")

    st.divider()

    # Input section
    st.subheader("📝 Enter Movie Review")
    
    samples = {
        "Positive Example": "This movie was absolutely amazing! The acting was impressive and the story kept me engaged.",
        "Negative Example": "Terrible movie. Predictable story and bad acting. Complete waste of time.",
        "Custom Review": ""
    }
    
    choice = st.selectbox("Choose sample or write your own:", samples.keys())
    user_input = st.text_area(
        "Review Text",
        value=samples[choice],
        height=150,
        placeholder="Write your review here..."
    )

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

    # Prediction
    if analyze_btn:
        if not user_input.strip():
            st.warning("⚠️ Please enter a review")
            return

        try:
            with st.spinner("🤖 Analyzing..."):
                result = predict_sentiment(user_input)
        except Exception as e:
            st.error(str(e))
            return

        # Display results
        st.divider()
        st.subheader("📊 Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", result["sentiment"])
        with col2:
            st.metric("Confidence", f"{result['confidence']*100:.2f}%")

        # Probability chart
        fig = go.Figure([
            go.Bar(
                x=["Negative", "Positive"],
                y=[result["negative_prob"]*100, result["positive_prob"]*100],
                text=[
                    f"{result['negative_prob']*100:.2f}%",
                    f"{result['positive_prob']*100:.2f}%"
                ],
                textposition="auto",
                marker_color=["#ff6b6b", "#51cf66"]
            )
        ])
        fig.update_layout(
            title="Sentiment Probability Distribution",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("Built with ❤️ using Streamlit & HuggingFace")


if __name__ == "__main__":
    main()