"""
FAQ CHATBOT BACKEND - NLTK 
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

# ==================== FAQ DATABASE ====================
FAQ_DATABASE = pd.read_csv('data.csv')

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
        
        # Get FAQ keywords
        faq_keywords = set(self.faq_df.iloc[faq_idx]['keywords'])
        
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
        
        # Prepare result
        result = {
            'faq_id': int(self.faq_df.iloc[best_idx]['id']),
            'question': self.faq_df.iloc[best_idx]['question'],
            'answer': self.faq_df.iloc[best_idx]['answer'],
            'category': self.faq_df.iloc[best_idx]['category'],
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
        return list(set(faq['category'] for faq in self.faq_data))
    
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
        "What time do you open?",
        "I forgot my password",
        "How can I pay?",
        "Track my package",
        "Return policy?"
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