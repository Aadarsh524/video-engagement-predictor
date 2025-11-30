import pandas as pd
import numpy as np
import re
import emoji
from textblob import TextBlob
from langdetect import detect, LangDetectException
from sklearn.decomposition import PCA

# ===========================
#  Top 50 Words
# ===========================
top_50_words = [
    '10', '2017', '2018', 'audio', 'avec', 'best', 'black', 'challenge', 'christmas',
    'clip', 'clip officiel', 'day', 'en', 'ep', 'episode', 'et', 'feat', 'food', 'ft',
    'game', 'game highlights', 'goals', 'hd', 'highlights', 'house', 'la', 'le',
    'life', 'live', 'love', 'music', 'nba', 'new', 'news', 'official',
    'official trailer', 'official video', 'officiel', 'real', 'season', 'songs',
    'star', 'time', 'trailer', 'trump', 'tv', 'video', 'vs', 'world', 'ðÿ'
]

# ===========================
#  Clickbait Phrases
# ===========================
clickbait_phrases = [
    'shocked', 'secret', 'never', 'always', 'top 10', "you won't believe",
    'going viral', 'this is why', 'what happened next', 'watch until end',
    'exposed', 'truth about', "they don't want you to know"
]

# ===========================
# 1. Text Cleaning
# ===========================
def clean_text(text, keep_emojis=True):
    if pd.isna(text):
        return ""
    text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    if keep_emojis:
        # Keep alphanumeric + spaces + emojis
        text = ''.join(ch for ch in text if ch.isalnum() or ch.isspace() or ch in emoji.EMOJI_DATA)
    else:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    return re.sub(r'\s+', ' ', text).strip()




# ===========================
# 3. Title Features
# ===========================
def extract_title_features(df, title_col='title'):
    df = df.copy()
    df['title_length'] = df[title_col].apply(len)
    df['uppercase_words'] = df[title_col].apply(lambda x: sum(1 for word in x.split() if word.isupper()))
    df['contains_numbers_or_emojis'] = df[title_col].apply(
        lambda x: int(bool(re.search(r'\d', x)) or any(c in emoji.EMOJI_DATA for c in x))
    )
    return df


# ===========================
# 4. Emoji Features
# ===========================
def extract_emoji_features(df, title_col='title_cleaned'):
    df = df.copy()
    df['num_emojis'] = df[title_col].apply(lambda x: len([c for c in str(x) if c in emoji.EMOJI_DATA]))
    df['has_emoji'] = (df['num_emojis'] > 0).astype(int)
    df['emojis_list'] = df[title_col].apply(lambda x: [c for c in str(x) if c in emoji.EMOJI_DATA])
    return df


# ===========================
# 5. Sentiment Features
# ===========================
def extract_sentiment_features(df, title_col='title'):
    df = df.copy()
    df['sentiment_polarity'] = df[title_col].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_subjectivity'] = df[title_col].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return df


# ===========================
# 6. Top-50 PCA + Boolean Flag
# ===========================
def compute_top50_features(df, cleaned_col='title_cleaned', top_words=None):
    df = df.copy()
    if top_words is None:
        raise ValueError("top_words list is required")

    top_set = set(top_words)


    # Bag-of-words vector
    def word_vector(text):
        words = set(str(text).split())
        return [1 if w in words else 0 for w in top_words]

    matrix = np.array(df[cleaned_col].apply(word_vector).tolist())

    # PCA (only if enough samples)
    if matrix.shape[0] < 3:
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





# ===========================
# 9. Title Word Count
# ===========================
def add_title_word_count(df, title_col='title_cleaned'):
    df = df.copy()
    df['title_word_count'] = df[title_col].apply(lambda x: len(str(x).split()))
    return df


# ===========================
# 10. Question Mark Feature
# ===========================
def add_has_question(df, title_col='title'):
    df = df.copy()
    df['has_question'] = df[title_col].apply(lambda x: int('?' in str(x)))
    return df




# ===========================
# 12. Clickbait Phrase Detection
# ===========================
def add_clickbait_feature(df, title_col='title'):
    df = df.copy()
    phrases = [p.lower() for p in clickbait_phrases]

    def check_clickbait(t):
        t = str(t).lower()
        return int(any(p in t for p in phrases))

    df['is_clickbait'] = df[title_col].apply(check_clickbait)
    return df


# ===========================
# 13. FULL PIPELINE (All Features)
# ===========================
def create_all_features(df, title_col='title'):
    print("Starting feature engineering pipeline...")

    # Clean text
    df['title_cleaned'] = df[title_col].apply(clean_text)

    # Title features
    df = extract_title_features(df, title_col)

    # Emoji features
    df = extract_emoji_features(df, title_col='title_cleaned')

    # Sentiment
    df = extract_sentiment_features(df, title_col)

    # Top-50 + PCA
    df = compute_top50_features(df, cleaned_col='title_cleaned', top_words=top_50_words)

    # Day of week numeric
    day_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    df['published_day_of_week_num'] = df['published_day_of_week'].str.lower().map(day_map)

    # NEW FEATURES


    df = add_title_word_count(df)
    df = add_has_question(df)
    df = add_clickbait_feature(df)

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
