"""
FAQ CHATBOT BACKEND - NLTK (FIXED VERSION)
================================
Clean backend implementation using only NLTK library
No spaCy dependency required

File: faq_backend.py

Installation:
    pip install nltk scikit-learn numpy pandas

Author: AI Engineering Intern
"""

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Tuple

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def load_faq_database():
    """Load FAQ database from CSV and convert keywords to list"""
    # Read CSV with explicit handling
    df = pd.read_csv('data.csv', encoding='utf-8')
    
    # Debug: Print what we loaded
    print("\n" + "="*60)
    print("LOADING FAQ DATABASE")
    print("="*60)
    print(f"CSV Columns: {df.columns.tolist()}")
    print(f"Total rows: {len(df)}")
    print("\nFirst row raw data:")
    for col in df.columns:
        print(f"  {col}: '{df.iloc[0][col]}'")
    
    # Rename faq_id to id
    if 'faq_id' in df.columns:
        df = df.rename(columns={'faq_id': 'id'})
    elif 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    
    # Convert keywords column (comma-separated string) to list
    if 'keywords' in df.columns:
        df['keywords'] = df['keywords'].apply(
            lambda x: [kw.strip().lower() for kw in str(x).split(',') if kw.strip()]
        )
    else:
        print("WARNING: No 'keywords' column found!")
        df['keywords'] = [[] for _ in range(len(df))]
    
    # Ensure all required columns exist
    required_cols = ['id', 'question', 'answer', 'keywords', 'category']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column: {col}")
            print(f"Available columns: {df.columns.tolist()}")
            raise ValueError(f"CSV missing required column: {col}")
    
    # Select only required columns in correct order
    df = df[required_cols].copy()
    
    # Debug: Print processed first record
    print("\nProcessed first FAQ record:")
    first_record = df.iloc[0].to_dict()
    for key, value in first_record.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    return df.to_dict('records')


FAQ_DATABASE = load_faq_database()


# ==================== TEXT PREPROCESSOR ====================
class TextPreprocessor:
    """
    Handle all text preprocessing using NLTK
    - Cleaning
    - Tokenization
    - Stop word removal
    - Lemmatization
    """
    
    def __init__(self):
        """Initialize preprocessor with NLTK tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing special characters and converting to lowercase
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        Remove common stop words
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Filtered list of tokens
        """
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their root form
        
        Args:
            tokens: List of word tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize(text)
        
        # Remove stop words
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize(tokens)
        
        # Join back to string
        return ' '.join(tokens)


# ==================== FAQ MATCHER ====================
class FAQMatcher:
    """
    Match user queries with FAQ database using:
    - TF-IDF + Cosine Similarity
    - Keyword Matching
    - Hybrid Scoring
    """
    
    def __init__(self, faq_data: List[Dict]):
        """
        Initialize matcher with FAQ database
        
        Args:
            faq_data: List of FAQ dictionaries
        """
        self.faq_df = pd.DataFrame(faq_data)
        self.preprocessor = TextPreprocessor()
        
        # Ensure 'id' column exists
        if 'id' not in self.faq_df.columns:
            self.faq_df.insert(0, 'id', range(1, len(self.faq_df) + 1))
        
        # Preprocess all FAQ questions
        self.faq_df['processed_question'] = self.faq_df['question'].apply(
            self.preprocessor.preprocess
        )
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.faq_df['processed_question']
        )
    
    def cosine_similarity_score(self, user_query: str) -> Tuple[int, float]:
        """
        Calculate cosine similarity between query and all FAQs
        
        Args:
            user_query: User's question
            
        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        # Preprocess query
        processed_query = self.preprocessor.preprocess(user_query)
        
        # Convert to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get best match
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        return best_idx, best_score
    
    def keyword_match_score(self, user_query: str, faq_idx: int) -> float:
        """
        Calculate keyword matching score using Jaccard similarity
        
        Args:
            user_query: User's question
            faq_idx: Index of FAQ to compare
            
        Returns:
            Keyword match score (0-1)
        """
        # Get query tokens
        query_tokens = set(self.preprocessor.preprocess(user_query).split())
        
        # Get FAQ keywords (convert to lowercase)
        faq_keywords = set([str(kw).lower() for kw in self.faq_df.iloc[faq_idx]['keywords']])
        
        # Calculate Jaccard similarity
        if len(query_tokens) == 0:
            return 0.0
        
        intersection = query_tokens.intersection(faq_keywords)
        union = query_tokens.union(faq_keywords)
        
        return len(intersection) / len(union) if len(union) > 0 else 0.0
    
    def hybrid_match(self, user_query: str, weights: Dict[str, float] = None) -> Dict:
        """
        Hybrid matching using weighted combination of methods
        
        Args:
            user_query: User's question
            weights: Dictionary with 'cosine' and 'keyword' weights
            
        Returns:
            Dictionary with match results
        """
        if weights is None:
            weights = {'cosine': 0.7, 'keyword': 0.3}
        
        # Get cosine similarity score
        cosine_idx, cosine_score = self.cosine_similarity_score(user_query)
        
        # Calculate scores for all FAQs
        combined_scores = []
        for i in range(len(self.faq_df)):
            # Get cosine score for this FAQ
            processed_query = self.preprocessor.preprocess(user_query)
            query_vector = self.vectorizer.transform([processed_query])
            cos_score = cosine_similarity(query_vector, self.tfidf_matrix[i])[0][0]
            
            # Get keyword score
            key_score = self.keyword_match_score(user_query, i)
            
            # Calculate weighted combination
            combined = (weights['cosine'] * cos_score + 
                       weights['keyword'] * key_score)
            combined_scores.append(combined)
        
        # Get best match
        best_idx = np.argmax(combined_scores)
        best_score = combined_scores[best_idx]
        
        # Get keyword score for best match
        keyword_score = self.keyword_match_score(user_query, best_idx)
        
        # Prepare result - safely convert types
        faq_row = self.faq_df.iloc[best_idx]
        
        # Debug print
        print(f"Best match index: {best_idx}")
        print(f"FAQ row data: {faq_row.to_dict()}")
        
        # Safely get ID - try multiple ways
        try:
            if 'id' in faq_row and pd.notna(faq_row['id']):
                faq_id = int(faq_row['id'])
            elif 'faq_id' in faq_row and pd.notna(faq_row['faq_id']):
                faq_id = int(faq_row['faq_id'])
            else:
                faq_id = best_idx + 1
        except (ValueError, TypeError):
            faq_id = best_idx + 1
            print(f"Warning: Could not convert ID to int, using index: {faq_id}")
        
        result = {
            'faq_id': faq_id,
            'question': str(faq_row['question']),
            'answer': str(faq_row['answer']),
            'category': str(faq_row['category']),
            'confidence': float(best_score),
            'cosine_score': float(cosine_similarity(
                self.vectorizer.transform([self.preprocessor.preprocess(user_query)]),
                self.tfidf_matrix[best_idx]
            )[0][0]),
            'keyword_score': float(keyword_score)
        }
        
        return result


# ==================== FAQ CHATBOT ====================
class FAQChatbot:
    """
    Main chatbot class for handling user queries
    """
    
    def __init__(self, faq_data: List[Dict] = None, confidence_threshold: float = 0.2):
        """
        Initialize chatbot
        
        Args:
            faq_data: List of FAQ dictionaries (uses default if None)
            confidence_threshold: Minimum confidence score to return answer
        """
        if faq_data is None:
            faq_data = FAQ_DATABASE
        
        self.faq_data = faq_data
        self.confidence_threshold = confidence_threshold
        self.matcher = FAQMatcher(faq_data)
        self.conversation_history = []
    
    def get_response(self, user_query: str) -> Dict:
        """
        Get chatbot response for user query
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with response data
        """
        # Validate input
        if not user_query or not user_query.strip():
            return {
                'success': False,
                'message': 'Please enter a valid question',
                'confidence': 0.0
            }
        
        # Find best match
        match = self.matcher.hybrid_match(user_query)
        
        # Debug: Print match result
        print(f"\n--- Query: '{user_query}' ---")
        print(f"Matched FAQ ID: {match.get('faq_id')}")
        print(f"Matched Question: {match.get('question')}")
        print(f"Answer (first 100 chars): {match.get('answer')[:100]}...")
        print(f"Confidence: {match.get('confidence'):.2%}")
        
        # Store in conversation history
        self.conversation_history.append({
            'query': user_query,
            'response': match,
            'timestamp': pd.Timestamp.now()
        })
        
        # Check confidence threshold
        if match['confidence'] < self.confidence_threshold:
            return {
                'success': False,
                'message': "I'm sorry, I couldn't find a relevant answer to your question. Please try rephrasing or contact our support team at support@company.com",
                'confidence': match['confidence'],
                'matched_question': match['question']
            }
        
        # Return successful response
        return {
            'success': True,
            'answer': match['answer'],
            'matched_question': match['question'],
            'category': match['category'],
            'confidence': match['confidence'],
            'cosine_score': match['cosine_score'],
            'keyword_score': match['keyword_score'],
            'faq_id': match['faq_id']
        }
    
    def get_all_faqs(self) -> List[Dict]:
        """Get all FAQs in database"""
        return self.faq_data
    
    def get_categories(self) -> List[str]:
        """Get unique categories"""
        categories = list(set(faq['category'] for faq in self.faq_data))
        return sorted(categories)
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def get_statistics(self) -> Dict:
        """Get chatbot statistics"""
        if not self.conversation_history:
            return {
                'total_queries': 0,
                'successful_matches': 0,
                'average_confidence': 0.0,
                'categories_used': []
            }
        
        successful = [h for h in self.conversation_history 
                     if h['response']['confidence'] >= self.confidence_threshold]
        
        avg_confidence = np.mean([h['response']['confidence'] 
                                 for h in self.conversation_history])
        
        categories = [h['response']['category'] for h in successful]
        
        return {
            'total_queries': len(self.conversation_history),
            'successful_matches': len(successful),
            'average_confidence': float(avg_confidence),
            'categories_used': list(set(categories))
        }


# ==================== INITIALIZE CHATBOT ====================
# Global chatbot instance
chatbot = FAQChatbot()


# ==================== UTILITY FUNCTIONS ====================
def get_chatbot() -> FAQChatbot:
    """Get chatbot instance"""
    return chatbot


if __name__ == "__main__":
    # Test the chatbot
    print("=" * 60)
    print("FAQ CHATBOT BACKEND TEST")
    print("=" * 60)
    
    test_queries = [
        "What programs are offered by the university?",
        "How can I apply for admission?",
        "What is the minimum eligibility for BS programs?",
        "Is entry test compulsory?",
        "What is the admission schedule?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chatbot.get_response(query)
        
        if response['success']:
            print(f"Answer: {response['answer']}")
            print(f"Confidence: {response['confidence']:.2%}")
            print(f"Category: {response['category']}")
        else:
            print(f"Message: {response['message']}")
            print(f"Confidence: {response['confidence']:.2%}")
        print("-" * 60)
    
    # Show statistics
    print("\nChatbot Statistics:")
    stats = chatbot.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")