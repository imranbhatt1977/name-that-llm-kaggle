# src/features.py
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

def extract_stylistic_features(text):
    """Extracts advanced stylistic, lexical, and punctuation features from a text passage."""
    text_str = str(text)
    char_count = len(text_str)
    words = text_str.split()
    word_count = len(words)
    
    if word_count == 0:
        return {
            'char_length': 0, 'word_count': 0, 'avg_word_length': 0,
            'vocab_richness': 0, 'stopword_ratio': 0,
            'comma_density': 0, 'period_density': 0, 'question_density': 0,
            'exclamation_density': 0, 'semicolon_density': 0, 'em_dash_density': 0,
            'uppercase_ratio': 0
        }
    
    # 1. Lexical & Vocabulary Metrics
    avg_word_length = sum(len(w) for w in words) / word_count
    vocab_richness = len(set(w.lower() for w in words)) / word_count  # Type-Token Ratio (TTR)
    
    # 2. Stopword Ratios
    stopword_count = sum(1 for w in words if w.lower() in STOP_WORDS)
    stopword_ratio = stopword_count / word_count
    
    # 3. Punctuation Densities (counts per 100 words)
    comma_count = text_str.count(',')
    period_count = text_str.count('.')
    question_count = text_str.count('?')
    exclamation_count = text_str.count('!')
    semicolon_count = text_str.count(';')
    em_dash_count = text_str.count('—') + text_str.count('--')
    
    scale = 100.0 / word_count
    
    # 4. Capitalization
    uppercase_count = sum(1 for c in text_str if c.isupper())
    uppercase_ratio = uppercase_count / max(char_count, 1)
    
    return {
        'char_length': char_count,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'vocab_richness': vocab_richness,
        'stopword_ratio': stopword_ratio,
        'comma_density': comma_count * scale,
        'period_density': period_count * scale,
        'question_density': question_count * scale,
        'exclamation_density': exclamation_count * scale,
        'semicolon_density': semicolon_count * scale,
        'em_dash_density': em_dash_count * scale,
        'uppercase_ratio': uppercase_ratio
    }

def process_dataframe_features(df):
    """Applies stylistic feature extraction across an entire dataframe."""
    feature_list = df['text'].apply(extract_stylistic_features)
    return pd.DataFrame(list(feature_list))
