# src/utils.py
import re
import emoji
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# -----------------------------
# Constants
# -----------------------------
CATEGORIES = {
    'Film & Animation': 1, 'Autos & Vehicles': 2, 'Music': 10, 'Pets & Animals': 15,
    'Sports': 17, 'Travel & Events': 19, 'Gaming': 20, 'People & Blogs': 22,
    'Comedy': 23, 'Entertainment': 24, 'News & Politics': 25, 'Howto & Style': 26,
    'Education': 27, 'Science & Technology': 28, 'Nonprofits & Activism': 29
}

DAYS = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

CLICKBAIT_PHRASES = [
    'shocked', 'secret', 'never', 'always', 'top 10', "you won't believe",
    'going viral', 'this is why', 'what happened next', 'watch until end',
    'exposed', 'truth about', "they don't want you to know"
]

# -----------------------------
# Helper Functions
# -----------------------------
def count_emojis(text):
    return len([c for c in str(text) if c in emoji.EMOJI_DATA])

def has_emoji(text):
    return 1 if count_emojis(text) > 0 else 0

def contains_numbers_or_emojis(text):
    return 1 if (re.search(r'\d', str(text)) or count_emojis(text) > 0) else 0

def count_uppercase_words(text):
    return sum(1 for w in str(text).split() if w.isupper() and len(w) > 1)

def has_question(text):
    return 1 if '?' in str(text) else 0

def is_clickbait(text):
    text_lower = str(text).lower()
    return 1 if any(phrase in text_lower for phrase in CLICKBAIT_PHRASES) else 0

# Placeholder sentiment function; replace with TextBlob or similar in training
def get_sentiment(text):
    return 0.0, 0.5

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features_from_title(title):
    polarity, subjectivity = get_sentiment(title)
    return {
        'title_length': len(title),
        'title_word_count': len(str(title).split()),
        'uppercase_words': count_uppercase_words(title),
        'num_emojis': count_emojis(title),
        'has_emoji': has_emoji(title),
        'contains_numbers_or_emojis': contains_numbers_or_emojis(title),
        'has_question': has_question(title),
        'is_clickbait': is_clickbait(title),
        'sentiment_polarity': polarity,
        'sentiment_subjectivity': subjectivity
    }

# -----------------------------
# Top-50 PCA features
# -----------------------------
def compute_top50_features(df, cleaned_col='title_cleaned', top_words=None):
    """
    Compute one-hot encoding of top 50 words, then reduce via PCA to 3 components
    and detect if title contains English words from top_words.
    """
    if top_words is None:
        raise ValueError("top_words list is required")
    df = df.copy()
    

    
    # Create word presence matrix
    words_list = [set(str(x).split()) for x in df[cleaned_col]]
    matrix = np.array([[1 if w in ws else 0 for w in top_words] for ws in words_list])
    
    # PCA to 3 components
    if matrix.shape[0] < 3:
        df['top50_pca1'] = 0.0
        df['top50_pca2'] = 0.0
        df['top50_pca3'] = 0.0
    else:
        pca_vals = PCA(n_components=3).fit_transform(matrix)
        df['top50_pca1'] = pca_vals[:, 0]
        df['top50_pca2'] = pca_vals[:, 1]
        df['top50_pca3'] = pca_vals[:, 2]
        
    return df
