"""
Feature Extraction Module for Video Engagement Predictor
Based on notebooks 03 and 04
"""

import pandas as pd
import numpy as np
import re
import emoji
from textblob import TextBlob
from langdetect import detect, LangDetectException


def clean_text(text, keep_emojis=True):
    """
    Clean text while preserving emojis
    Based on your get_plain_text function from notebook 04
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    if keep_emojis:
        # Keep alphanumeric, spaces, and emojis
        text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in emoji.EMOJI_DATA)
    else:
        # Remove all non-alphanumeric except spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def detect_language(text):
    """Detect language with better error handling"""
    text = str(text).strip()
    if not text or len(text.split()) < 2:
        return 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'unknown'


def extract_title_features(df, title_col='title'):
    """
    Extract features from video titles
    Based on your notebook 03 implementation
    """
    df = df.copy()
    
    # Basic length features
    df['title_length'] = df[title_col].apply(len)
    
    # Uppercase words (your implementation)
    df['uppercase_words'] = df[title_col].apply(
        lambda x: sum(1 for word in x.split() if word.isupper())
    )
    
    # Combined numbers/emojis check (your has_number_or_emojis function)
    def has_number_or_emojis(text):
        has_number = bool(re.search(r"\d", text))
        has_emoji = any(char in emoji.EMOJI_DATA for char in text)
        return int(has_number or has_emoji)
    
    df['contains_numbers_or_emojis'] = df[title_col].apply(has_number_or_emojis)
    
    return df


def extract_emoji_features(df, title_col='title_cleaned'):
    """Extract emoji-related features"""
    df = df.copy()
    
    def count_emojis(text):
        return len([c for c in str(text) if c in emoji.EMOJI_DATA])
    
    def extract_emojis(text):
        return [c for c in str(text) if c in emoji.EMOJI_DATA]
    
    # Basic emoji features
    df['num_emojis'] = df[title_col].apply(count_emojis)
    df['has_emoji'] = (df['num_emojis'] > 0).astype(int)
    
    # Store emoji list for analysis (optional)
    df['emojis_list'] = df[title_col].apply(extract_emojis)
    
    return df



def extract_sentiment_features(df, title_col='title'):
    """
    Extract sentiment features using TextBlob
    Based on your notebook 03 implementation
    """
    df = df.copy()
    
    # Sentiment polarity
    df['sentiment_polarity'] = df[title_col].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    
    # Sentiment subjectivity
    df['sentiment_subjectivity'] = df[title_col].apply(
        lambda x: TextBlob(x).sentiment.subjectivity
    )
    
    return df




def filter_english_only(df, title_col='title_cleaned'):
    """
    Filter for English language titles of videos
    """
    df = df.copy()
    
    def is_english(text):
        text = str(text).strip()
        if not text or len(text.split()) < 2:
            return False
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False
    
    # Apply filter
    df['is_english'] = df[title_col].apply(is_english)
    df_filtered = df[df['is_english']].copy()
    df_filtered = df_filtered.reset_index(drop=True)
    
    print(f"Original dataset: {len(df)} videos")
    print(f"English-only dataset: {len(df_filtered)} videos")
    print(f"Filtered out: {len(df) - len(df_filtered)} videos")
    
    return df_filtered



def create_all_features(df, title_col='title', filter_english=True):
    
    print("Starting feature engineering pipeline...")
    print(f"Initial dataset size: {df.shape}")
    
    # Step 1: Clean text
    print("\n[1/5] Cleaning text...")
    df['title_cleaned'] = df[title_col].apply(clean_text)
    
    # Step 2: Filter English (optional)
    if filter_english:
        print("\n[2/5] Filtering English-only videos...")
        df = filter_english_only(df, 'title_cleaned')
    
    # Step 3: Extract title features
    print("\n[3/5] Extracting title features...")
    df = extract_title_features(df, title_col)
    
    # Step 4: Extract emoji features
    print("\n[4/5] Extracting emoji features...")
    df = extract_emoji_features(df, title_col)
    
    # Step 5: Extract sentiment features
    print("\n[5/5] Extracting sentiment features...")
    df = extract_sentiment_features(df, title_col)
    
    print("\n✅ Feature engineering complete!")
    print(f"Final dataset size: {df.shape}")
    print(f"Total features created: {len([col for col in df.columns if col not in ['title', 'title_cleaned']])}")
    
    return df



def get_feature_summary(df):
    """Print summary of created features"""
    feature_cols = [col for col in df.columns if col not in ['title', 'title_cleaned', 'emojis_list']]
    
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    for col in feature_cols:
        print(f"\n{col}:")
        print(f"  Type: {df[col].dtype}")
        print(f"  Missing: {df[col].isnull().sum()}")
        if df[col].dtype in ['int64', 'float64']:
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Range: [{df[col].min():.2f}, {df[col].max():.2f}]")



