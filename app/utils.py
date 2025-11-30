import re, emoji
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

CATEGORIES = {
    'Film & Animation': 1, 'Autos & Vehicles': 2, 'Music': 10, 'Pets & Animals': 15,
    'Sports': 17, 'Travel & Events': 19, 'Gaming': 20, 'People & Blogs': 22,
    'Comedy': 23, 'Entertainment': 24, 'News & Politics': 25, 'Howto & Style': 26,
    'Education': 27, 'Science & Technology': 28, 'Nonprofits & Activism': 29,
}

DAYS = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5,'Sunday':6}

def count_emojis(text): return len([c for c in text if c in emoji.EMOJI_DATA])
def has_emoji(text): return 1 if count_emojis(text) > 0 else 0
def count_uppercase_words(text): return sum(1 for w in str(text).split() if w.isupper() and len(w)>1)
def contains_numbers_or_emojis(text): return 1 if (bool(re.search(r'\d', str(text))) or has_emoji(text)) else 0
def get_sentiment(text): return 0.0, 0.5  # placeholder
def extract_features_from_title(title):
    polarity, subjectivity = get_sentiment(title)
    return {
        'title_length': len(title), 'uppercase_words': count_uppercase_words(title),
        'num_emojis': count_emojis(title), 'has_emoji': has_emoji(title),
        'contains_numbers_or_emojis': contains_numbers_or_emojis(title),
        'sentiment_polarity': polarity, 'sentiment_subjectivity': subjectivity
    }

def is_title_english(text, top_words): return 1 if any(w in set(str(text).split()) for w in top_words) else 0

def compute_top50_features(df, cleaned_col='title_cleaned', top_words=None):
    df = df.copy()
    if top_words is None: raise ValueError("top_words list is required")
    df['is_title_english'] = df[cleaned_col].apply(lambda x: is_title_english(x, top_words))
    
    def word_vector(text): return [1 if w in set(str(text).split()) else 0 for w in top_words]
    matrix = np.array(df[cleaned_col].apply(word_vector).tolist())
    
    if matrix.shape[0] < 3:
        df['top50_pca1'] = 0; df['top50_pca2'] = 0; df['top50_pca3'] = 0
    else:
        pca_values = PCA(n_components=3).fit_transform(matrix)
        df['top50_pca1'] = pca_values[:,0]; df['top50_pca2'] = pca_values[:,1]; df['top50_pca3'] = pca_values[:,2]
    return df
