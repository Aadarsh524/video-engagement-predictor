import pandas as pd
import numpy as np
import re
import emoji
from textblob import TextBlob
from langdetect import detect, LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# --- Top 50 words list (must be comma-separated!) ---
top_50_words = [
    '10', '2017', '2018', 'audio', 'avec', 'best', 'black', 'challenge', 'christmas',
    'clip', 'clip officiel', 'day', 'en', 'ep', 'episode', 'et', 'feat', 'food', 'ft',
    'game', 'game highlights', 'goals', 'hd', 'highlights', 'house', 'la', 'le',
    'life', 'live', 'love', 'music', 'nba', 'new', 'news', 'official',
    'official trailer', 'official video', 'officiel', 'real', 'season', 'songs',
    'star', 'time', 'trailer', 'trump', 'tv', 'video', 'vs', 'world', 'ðÿ'
]

# --- 1. Text Cleaning ---
def clean_text(text, keep_emojis=True):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    if keep_emojis:
        text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in emoji.EMOJI_DATA)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

# --- 2. Title Language Flag (boolean) ---
def is_title_english(text, top_words):
    words = set(str(text).split())
    num_top_words = len(words & set(top_words))
    return num_top_words >= len(top_words) / 2  # majority of top words

# --- 3. Title Features ---
def extract_title_features(df, title_col='title'):
    df = df.copy()
    df['title_length'] = df[title_col].apply(len)
    df['uppercase_words'] = df[title_col].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
    df['contains_numbers_or_emojis'] = df[title_col].apply(lambda x: int(bool(re.search(r'\d', x)) or any(c in emoji.EMOJI_DATA for c in x)))
    return df

# --- 4. Emoji Features ---
def extract_emoji_features(df, title_col='title_cleaned'):
    df = df.copy()
    df['num_emojis'] = df[title_col].apply(lambda x: len([c for c in str(x) if c in emoji.EMOJI_DATA]))
    df['has_emoji'] = (df['num_emojis'] > 0).astype(int)
    df['emojis_list'] = df[title_col].apply(lambda x: [c for c in str(x) if c in emoji.EMOJI_DATA])
    return df

# --- 5. Sentiment Features ---
def extract_sentiment_features(df, title_col='title'):
    df = df.copy()
    df['sentiment_polarity'] = df[title_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df[title_col].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df

# --- 6. Top-50 Score & PCA Features ---
def compute_top50_features(df, cleaned_col='title_cleaned', top_words=None):
    df = df.copy()
    if top_words is None:
        raise ValueError("top_words list is required")

    top_set = set(top_words)

    # --- Boolean flag ---
    df['is_title_english'] = df[cleaned_col].apply(lambda x: is_title_english(x, top_words))

    
    # --- PCA on top-50 words ---
    def word_vector(text):
        words = set(str(text).split())
        return [1 if w in words else 0 for w in top_words]

    matrix = np.array(df[cleaned_col].apply(word_vector).tolist())

    if matrix.shape[0] < 3:
        # Not enough samples for PCA
        df['top50_pca1'] = 0
        df['top50_pca2'] = 0
        df['top50_pca3'] = 0
    else:
        pca = PCA(n_components=3)
        pca_values = pca.fit_transform(matrix)
        df['top50_pca1'] = pca_values[:, 0]
        df['top50_pca2'] = pca_values[:, 1]
        df['top50_pca3'] = pca_values[:, 2]

    return df

# --- 7. Full Pipeline ---
def create_all_features(df, title_col='title'):
    print("Starting feature engineering pipeline...")
    df['title_cleaned'] = df[title_col].apply(clean_text)
    df = extract_title_features(df, title_col)
    df = extract_emoji_features(df, title_col='title_cleaned')
    df = extract_sentiment_features(df, title_col=title_col)
    df = compute_top50_features(df, cleaned_col='title_cleaned', top_words=top_50_words)

    # --- Map day of week to number ---
    day_map = {
        'monday':0,'tuesday':1,'wednesday':2,'thursday':3,'friday':4,'saturday':5,'sunday':6
    }
    df['published_day_of_week_num'] = df['published_day_of_week'].str.lower().map(day_map)

    print("Feature engineering complete!")
    return df

# --- 8. Feature Summary ---
def get_feature_summary(df):
    feature_cols = [col for col in df.columns if col not in ['title', 'title_cleaned', 'emojis_list']]
    print("\nFeature Summary:")
    for col in feature_cols:
        print(f"{col}: Type={df[col].dtype}, Missing={df[col].isnull().sum()}")
        if df[col].dtype in ['int64','float64']:
            print(f"   Mean={df[col].mean():.2f}, Range=[{df[col].min():.2f},{df[col].max():.2f}]")
