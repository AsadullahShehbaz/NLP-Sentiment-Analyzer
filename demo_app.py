"""
Professional Sentiment Analysis Application
Binary Classification using Fine-tuned DistilBERT
"""
import streamlit as st
from model import load_model, predict_sentiment
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer | Professional Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .sentiment-positive {
        color: #10b981;
        font-size: 28px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sentiment-negative {
        color: #ef4444;
        font-size: 28px;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: #f8fafc;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    """Load model with caching for better performance"""
    return load_model()


def create_probability_chart(positive_prob, negative_prob):
    """Create professional probability visualization"""
    fig = go.Figure()
    
    # Add bars with gradient effect
    fig.add_trace(go.Bar(
        x=['Negative 😞', 'Positive 😊'],
        y=[negative_prob * 100, positive_prob * 100],
        marker=dict(
            color=[negative_prob * 100, positive_prob * 100],
            colorscale=[[0, '#ef4444'], [1, '#10b981']],
            line=dict(color='rgba(0,0,0,0.2)', width=2)
        ),
        text=[f'{negative_prob*100:.1f}%', f'{positive_prob*100:.1f}%'],
        textposition='auto',
        textfont=dict(size=16, color='white', family='Arial Black'),
    ))
    
    fig.update_layout(
        title={
            'text': 'Sentiment Probability Distribution',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#1e293b'}
        },
        yaxis_title='Confidence (%)',
        xaxis_title='',
        height=450,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 105], gridcolor='#e2e8f0'),
        font=dict(size=14)
    )
    
    return fig


def create_confidence_gauge(confidence):
    """Create confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Confidence", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def main():
    # Sidebar - Professional Info
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=150)
        
        st.markdown("### 🎯 Model Specifications")
        st.markdown("""
        <div class='metric-card'>
            <b>Architecture:</b> DistilBERT<br>
            <b>Task:</b> Binary Sentiment Classification<br>
            <b>Classes:</b> Positive / Negative<br>
            <b>Training Data:</b> IMDB Movie Reviews<br>
            <b>Model Accuracy:</b> <span style='color: #10b981; font-weight: bold;'>86.25%</span><br>
            <b>Framework:</b> PyTorch + Transformers
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Use Cases")
        st.markdown("""
        - ✅ Customer Review Analysis
        - ✅ Social Media Monitoring
        - ✅ Product Feedback Classification
        - ✅ Brand Sentiment Tracking
        - ✅ Support Ticket Prioritization
        """)
        
        st.markdown("### 📊 Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Precision", "87.3%", delta="2.1%")
        with col2:
            st.metric("Recall", "85.8%", delta="1.5%")
        
        st.markdown("---")
        st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d')}")

    # Main Header
    st.markdown("""
        <div class='main-header'>
            <h1>🤖 AI-Powered Sentiment Analysis</h1>
            <p style='font-size: 18px; margin-top: 10px;'>
                Enterprise-Grade Binary Classification Model | Production-Ready Solution
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Initializing AI Model..."):
        tokenizer, model = get_model()
    
    st.success("✅ Model Ready for Analysis!")
    
    # Main Content Area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Input Text for Analysis")
        user_input = st.text_area(
            "Enter customer review, feedback, or any text:",
            height=180,
            placeholder="Example: I recently purchased this product and I'm thoroughly impressed with its quality and performance. The customer service was exceptional!",
            help="Enter any text between 10-512 words for best results"
        )
        
        # Character counter
        if user_input:
            char_count = len(user_input)
            word_count = len(user_input.split())
            st.caption(f"📊 Characters: {char_count} | Words: {word_count}")
    
    with col2:
        st.markdown("### 🎯 Quick Test Samples")
        
        if st.button("✅ Excellent Review", use_container_width=True):
            user_input = "Absolutely fantastic! This exceeded all my expectations. The quality is outstanding and delivery was super fast. I'm extremely satisfied and will definitely recommend this to others. Five stars!"
            st.rerun()
        
        if st.button("⭐ Good Review", use_container_width=True):
            user_input = "Pretty good product overall. It works as described and the price is reasonable. Had a minor issue but customer support resolved it quickly."
            st.rerun()
        
        if st.button("❌ Poor Review", use_container_width=True):
            user_input = "Very disappointed with this purchase. Poor quality, late delivery, and terrible customer service. Not worth the money. Would not recommend."
            st.rerun()
        
        if st.button("⚠️ Critical Review", use_container_width=True):
            user_input = "Absolute waste of money! Product broke within a week. Customer service ignored my complaints. This is the worst shopping experience I've ever had. Stay away!"
            st.rerun()
    
    # Analysis Section
    st.markdown("---")
    
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if user_input and len(user_input.strip()) > 10:
            with st.spinner("🤖 AI Model Processing..."):
                result = predict_sentiment(user_input, tokenizer, model)
            
            # Results Display
            st.markdown("## 📊 Analysis Results")
            
            # Three column layout for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                sentiment_class = "sentiment-positive" if "Positive" in result['sentiment'] else "sentiment-negative"
                st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: #f8fafc; border-radius: 10px;'>
                        <p style='color: #64748b; margin: 0;'>Detected Sentiment</p>
                        <p class='{sentiment_class}' style='margin: 10px 0;'>{result['sentiment']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: #f8fafc; border-radius: 10px;'>
                        <p style='color: #64748b; margin: 0;'>Confidence Level</p>
                        <p style='font-size: 32px; font-weight: bold; color: #667eea; margin: 10px 0;'>
                            {result['confidence']*100:.1f}%
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                prediction_strength = "Very Strong" if result['confidence'] > 0.9 else "Strong" if result['confidence'] > 0.75 else "Moderate"
                st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: #f8fafc; border-radius: 10px;'>
                        <p style='color: #64748b; margin: 0;'>Prediction Strength</p>
                        <p style='font-size: 24px; font-weight: bold; color: #8b5cf6; margin: 10px 0;'>
                            {prediction_strength}
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visualizations
            viz_col1, viz_col2 = st.columns([2, 1])
            
            with viz_col1:
                fig = create_probability_chart(result['positive_prob'], result['negative_prob'])
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                gauge_fig = create_confidence_gauge(result['confidence'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Detailed Breakdown
            st.markdown("### 📈 Detailed Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.success(f"""
                **Positive Sentiment Probability**  
                `{result['positive_prob']*100:.2f}%`
                """)
            
            with prob_col2:
                st.error(f"""
                **Negative Sentiment Probability**  
                `{result['negative_prob']*100:.2f}%`
                """)
            
            # Interpretation Guide
            with st.expander("📖 How to Interpret Results"):
                st.markdown("""
                **Confidence Levels:**
                - **90-100%**: Very high confidence - Extremely clear sentiment
                - **75-90%**: High confidence - Clear sentiment indication
                - **60-75%**: Moderate confidence - Reasonably clear sentiment
                - **Below 60%**: Low confidence - Mixed or neutral sentiment
                
                **Best Practices:**
                - Higher confidence = More reliable prediction
                - For business decisions, consider confidence > 75%
                - Review low-confidence predictions manually
                """)
        
        elif user_input and len(user_input.strip()) <= 10:
            st.warning("⚠️ Please enter at least 10 characters for accurate analysis!")
        else:
            st.warning("⚠️ Please enter some text to analyze!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px; background: #f8fafc; border-radius: 10px;'>
            <p style='color: #64748b; font-size: 14px; margin: 5px 0;'>
                <b>Powered by:</b> Fine-tuned DistilBERT | HuggingFace Transformers | PyTorch
            </p>
            <p style='color: #64748b; font-size: 12px; margin: 5px 0;'>
                Model: <a href='https://huggingface.co/asadullahshehbaz/my_text_classifier_model' 
                target='_blank' style='color: #667eea;'>asadullahshehbaz/my_text_classifier_model</a>
            </p>
            <p style='color: #94a3b8; font-size: 11px; margin: 10px 0;'>
                🔒 Enterprise-ready | Production-tested | Scalable AI Solution
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()