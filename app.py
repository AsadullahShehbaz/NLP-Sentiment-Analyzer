"""
STREAMLIT FRONTEND FOR FAQ CHATBOT (FIXED)
===================================
Beautiful interactive UI that connects to FastAPI backend

File: app.py

Installation:
    pip install streamlit requests plotly pandas

Run Command:
    streamlit run app.py

Make sure FastAPI backend is running first:
    uvicorn main:app --reload

Author: AI Engineering Intern
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ==================== CONFIGURATION ====================
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat messages */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .bot-message {
        background: white;
        color: #333;
        padding: 15px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Stats cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        color: #667eea;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Confidence badges */
    .confidence-high {
        background: #10b981;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 12px;
    }
    
    .confidence-medium {
        background: #f59e0b;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 12px;
    }
    
    .confidence-low {
        background: #ef4444;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 12px;
    }
    
    /* Category badge */
    .category-badge {
        background: #667eea;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 11px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 30px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.3);
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    h2, h3 {
        color: white;
    }
    
    /* Input field */
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            'role': 'bot',
            'content': '👋 Hello! I\'m your AI FAQ Assistant. Ask me anything about our services!',
            'timestamp': datetime.now().strftime('%H:%M:%S')
        }
    ]

if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

# ==================== API FUNCTIONS ====================

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def send_query(query):
    """Send query to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={"query": query},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {
            "success": False,
            "message": f"API Error: {e.response.text if hasattr(e, 'response') else str(e)}",
            "confidence": 0.0
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error connecting to API: {str(e)}",
            "confidence": 0.0
        }


def get_all_faqs():
    """Get all FAQs from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/faqs", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error loading FAQs: {str(e)}")
        return []


def get_statistics():
    """Get chatbot statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/statistics", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")
        return None


def get_categories():
    """Get all categories from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/categories", timeout=5)
        response.raise_for_status()
        data = response.json()
        # Ensure it's a list
        if isinstance(data, list):
            return data
        else:
            st.error(f"Unexpected categories format: {type(data)}")
            return []
    except Exception as e:
        st.error(f"Error loading categories: {str(e)}")
        return []


# ==================== HELPER FUNCTIONS ====================

def get_confidence_badge(confidence):
    """Generate confidence badge HTML"""
    if confidence >= 0.7:
        return f'<span class="confidence-high">🟢 High: {confidence:.0%}</span>'
    elif confidence >= 0.4:
        return f'<span class="confidence-medium">🟡 Medium: {confidence:.0%}</span>'
    else:
        return f'<span class="confidence-low">🔴 Low: {confidence:.0%}</span>'


def display_message(role, content, timestamp, metadata=None):
    """Display a chat message"""
    if role == 'user':
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong> <small style="opacity: 0.7;">({timestamp})</small><br/>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        bot_content = f"""
        <div class="bot-message">
            <strong>🤖 AI Assistant</strong> <small style="opacity: 0.7;">({timestamp})</small><br/>
            {content}
        """
        
        if metadata:
            if metadata.get('category'):
                bot_content += f'<br/><span class="category-badge">📁 {metadata["category"]}</span>'
            
            if metadata.get('confidence') is not None:
                bot_content += f'<br/>{get_confidence_badge(metadata["confidence"])}'
            
            if metadata.get('matched_question'):
                bot_content += f'<br/><small style="opacity: 0.7;">💡 Matched: "{metadata["matched_question"]}"</small>'
        
        bot_content += "</div>"
        st.markdown(bot_content, unsafe_allow_html=True)


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("# ⚙️ Settings")
    
    # API Health Check
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.warning("Please start the FastAPI server first:\n```uvicorn main:app --reload```")
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📊 Statistics", "📚 FAQ Database", "ℹ️ About"]
    )
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = [
            {
                'role': 'bot',
                'content': '👋 Chat cleared! How can I help you?',
                'timestamp': datetime.now().strftime('%H:%M:%S')
            }
        ]
        st.rerun()
    
    if st.button("🔄 Refresh Statistics"):
        st.rerun()
    
    st.markdown("---")
    
    # Sample Questions
    st.markdown("### 💡 Try asking:")
    sample_questions = [
        "What programs are offered?",
        "What is the admission schedule?",
        "Is entry test compulsory?",
        "How can I apply for admission?",
        "What is the minimum eligibility?"
    ]
    
    for idx, question in enumerate(sample_questions):
        if st.button(question, key=f"sample_{idx}"):
            st.session_state.sample_query = question

# ==================== MAIN CONTENT ====================

# Page: Chat
if page == "💬 Chat":
    st.title("🤖 AI FAQ Chatbot")
    st.markdown("### Ask me anything about our services!")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            display_message(
                message['role'],
                message['content'],
                message['timestamp'],
                message.get('metadata')
            )
    
    # Check if sample question was clicked
    if 'sample_query' in st.session_state:
        user_input = st.session_state.sample_query
        del st.session_state.sample_query
    else:
        user_input = None
    
    # Chat input
    st.markdown("---")
    col1, col2 = st.columns([6, 1])
    
    with col1:
        query = st.text_input(
            "Your question:",
            placeholder="Type your question here...",
            label_visibility="collapsed",
            key="query_input",
            value=user_input if user_input else ""
        )
    
    with col2:
        send_button = st.button("📤 Send", use_container_width=True)
    
    # Process query
    if (send_button or user_input) and query:
        # Add user message
        timestamp = datetime.now().strftime('%H:%M:%S')
        st.session_state.messages.append({
            'role': 'user',
            'content': query,
            'timestamp': timestamp
        })
        
        # Show typing indicator
        with st.spinner('🤔 Thinking...'):
            time.sleep(0.5)
            
            # Get response from API
            response = send_query(query)
            
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            if response.get('success'):
                # Add bot message with metadata
                st.session_state.messages.append({
                    'role': 'bot',
                    'content': response['answer'],
                    'timestamp': timestamp,
                    'metadata': {
                        'confidence': response.get('confidence', 0),
                        'category': response.get('category'),
                        'matched_question': response.get('matched_question')
                    }
                })
            else:
                # Add error message
                st.session_state.messages.append({
                    'role': 'bot',
                    'content': response.get('message', 'Sorry, something went wrong.'),
                    'timestamp': timestamp,
                    'metadata': {
                        'confidence': response.get('confidence', 0)
                    }
                })
            
            st.session_state.query_count += 1
        
        st.rerun()

# Page: Statistics
elif page == "📊 Statistics":
    st.title("📊 Chatbot Statistics")
    
    if api_status:
        stats = get_statistics()
        
        if stats:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Queries</div>
                    <div class="metric-value">{stats.get('total_queries', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Successful Matches</div>
                    <div class="metric-value">{stats.get('successful_matches', 0)}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total = stats.get('total_queries', 0)
                success = stats.get('successful_matches', 0)
                success_rate = (success / total * 100) if total > 0 else 0
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Success Rate</div>
                    <div class="metric-value">{success_rate:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Confidence</div>
                    <div class="metric-value">{stats.get('average_confidence', 0):.0%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Charts
            if stats.get('categories_used'):
                st.markdown("### 📈 Categories Usage")
                
                # Category distribution
                category_counts = pd.Series(stats['categories_used']).value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Query Distribution by Category",
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No statistics available yet. Start chatting to generate data!")
    else:
        st.error("Cannot load statistics. API is not connected.")

# Page: FAQ Database
elif page == "📚 FAQ Database":
    st.title("📚 FAQ Database")
    
    if api_status:
        faqs = get_all_faqs()
        categories = get_categories()
        
        if faqs and categories:
            # Filter by category
            st.markdown("### 🔍 Filter by Category")
            selected_category = st.selectbox(
                "Select Category",
                ["All"] + list(categories),  # Ensure categories is a list
                label_visibility="collapsed"
            )
            
            # Display FAQs
            if selected_category != "All":
                faqs = [faq for faq in faqs if faq.get('category') == selected_category]
            
            st.markdown(f"### Showing {len(faqs)} FAQs")
            
            for faq in faqs:
                with st.expander(f"❓ {faq.get('question', 'N/A')}", expanded=False):
                    st.markdown(f"**Category:** `{faq.get('category', 'N/A')}`")
                    st.markdown(f"**Answer:** {faq.get('answer', 'N/A')}")
                    keywords = faq.get('keywords', [])
                    if isinstance(keywords, list):
                        st.markdown(f"**Keywords:** {', '.join(keywords)}")
                    else:
                        st.markdown(f"**Keywords:** {keywords}")
                    st.markdown(f"**ID:** {faq.get('id', 'N/A')}")
        else:
            st.warning("No FAQs available or unable to load categories.")
    else:
        st.error("Cannot load FAQs. API is not connected.")

# Page: About
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    
    st.markdown("""
    ## 🤖 AI FAQ Chatbot System
    
    ### 📝 Project Overview
    This is an NLP-based FAQ chatbot system built as part of an AI Engineering internship.
    It uses Natural Language Processing to understand user questions and match them with
    the most relevant FAQ from the database.
    
    ### 🛠️ Technology Stack
    
    **Backend:**
    - 🐍 Python 3.8+
    - 🤗 NLTK (Natural Language Toolkit)
    - 📊 scikit-learn (TF-IDF, Cosine Similarity)
    - ⚡ FastAPI (REST API)
    - 🔢 NumPy & Pandas
    
    **Frontend:**
    - 🎨 Streamlit (Interactive UI)
    - 📈 Plotly (Data Visualization)
    - 🎭 Custom CSS Styling
    
    ### 🧠 NLP Techniques Used
    
    1. **Text Preprocessing**
       - Tokenization
       - Stop word removal
       - Lemmatization
       - Text cleaning
    
    2. **Feature Engineering**
       - TF-IDF Vectorization
       - Keyword extraction
    
    3. **Similarity Matching**
       - Cosine Similarity
       - Keyword Matching
       - Hybrid Scoring
    
    ### 📊 System Architecture
    
    ```
    User Interface (Streamlit)
           ↓
    REST API (FastAPI)
           ↓
    NLP Backend (NLTK)
           ↓
    FAQ Database
    ```
    
    ### 🎯 Features
    
    ✅ Real-time query processing  
    ✅ Confidence scoring  
    ✅ Category classification  
    ✅ Conversation history  
    ✅ Statistics dashboard  
    ✅ Interactive FAQ browser  
    ✅ Sample question suggestions  
    
    ### 👨‍💻 Author : Asadullah Shehbaz 
    Built with ❤️ using Python
    
    ### 📚 Learning Resources
    - NLTK Documentation
    - FastAPI Tutorial
    - Streamlit Gallery
    - scikit-learn User Guide
    
    ---
    
    **Version:** 1.0.0  
    **Last Updated:** December 2025
                
    """)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; opacity: 0.8;'>
    <p>🤖 AI FAQ Chatbot | Built with Python, NLTK, FastAPI & Streamlit</p>
    <p>💡 AI Engineering Internship Project 2025</p>
</div>
""", unsafe_allow_html=True)