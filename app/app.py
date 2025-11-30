# app.py
import streamlit as st
import numpy as np
import pandas as pd
from model_loader import ModelLoader
from utils import CATEGORIES, DAYS, extract_features_from_title, compute_top50_features
from pathlib import Path

# Top-50 words (same as features.py)
TOP_50_WORDS = [
    '10','2017','2018','audio','avec','best','black','challenge','christmas',
    'clip','clip officiel','day','en','ep','episode','et','feat','food','ft',
    'game','game highlights','goals','hd','highlights','house','la','le','life',
    'live','love','music','nba','new','news','official','official trailer',
    'official video','officiel','real','season','songs','star','time','trailer',
    'trump','tv','video','vs','world','ðÿ'
]

st.set_page_config(page_title="YouTube Views Predictor", page_icon="📺", layout="wide")
st.title("📺 YouTube Video Views Predictor")

# Load models (cache)
@st.cache_resource
def load_models():
    loader = ModelLoader(
        pytorch_model_path="models/multimodal_pytorch.pth",
        xgboost_model_path="models/xgboost_tuned.pkl"
    )
    return loader

model_loader = load_models()

# Sidebar info
st.sidebar.title("Model Info")
st.sidebar.markdown("**PyTorch Multimodal** — Tabular + Title embeddings")
st.sidebar.markdown("Enter realistic values for best predictions")

# Inputs
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Video Details")
    title = st.text_input("Title", value="10 Amazing Python Tips for Beginners 🐍")
    category = st.selectbox("Category", options=list(CATEGORIES.keys()), index=list(CATEGORIES.keys()).index('Education'))
    publish_day = st.selectbox("Publish Day", options=list(DAYS.keys()))
    days_until_trending = st.number_input("Days Until Trending", min_value=0, max_value=365, value=2)
    comments_disabled = st.checkbox("Comments Disabled", value=False)
    ratings_disabled = st.checkbox("Ratings Disabled", value=False)

# Feature extraction
title_cleaned = pd.DataFrame({'title_cleaned': [title.lower()]})
title_top50 = compute_top50_features(title_cleaned, cleaned_col='title_cleaned', top_words=TOP_50_WORDS)
title_feats = extract_features_from_title(title)

# Clickbait flag
is_clickbait = int(any(p in title.lower() for p in [
    'shocked', 'secret', 'never', 'always', 'top 10', "you won't believe",
    'going viral', 'this is why', 'what happened next', 'watch until end',
    'exposed', 'truth about', "they don't want you to know"
]))

# Assemble features dict (keys must match tabular_cols)
features = {
    'title_length': title_feats['title_length'],
    'title_word_count': len(title.split()),
    'uppercase_words': title_feats['uppercase_words'],
    'num_emojis': title_feats['num_emojis'],
    'has_emoji': title_feats['has_emoji'],
    'contains_numbers_or_emojis': title_feats['contains_numbers_or_emojis'],
    'has_question': int('?' in title),
    'is_clickbait': is_clickbait,
    'sentiment_polarity': title_feats.get('sentiment_polarity', 0.0),
    'sentiment_subjectivity': title_feats.get('sentiment_subjectivity', 0.5),
    'top50_pca1': float(title_top50['top50_pca1'].iloc[0]),
    'top50_pca2': float(title_top50['top50_pca2'].iloc[0]),
    'top50_pca3': float(title_top50['top50_pca3'].iloc[0]),
    'is_published_weekend': int(DAYS[publish_day] >= 4),
    'category_id': int(CATEGORIES[category]),
    'comments_disabled': int(comments_disabled),
    'ratings_disabled': int(ratings_disabled),
    'days_until_trending': int(days_until_trending)
}

# Prediction UI
with col2:
    st.subheader("Prediction")
    if st.button("Predict Views"):
        with st.spinner("Running model..."):
            try:
                pred = model_loader.predict_pytorch(features, title)
                xgb_pred = model_loader.predict_xgboost(features)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                raise

            st.metric("Predicted Views (Multimodal)", f"{int(pred):,}")
            if xgb_pred:
                st.metric("XGBoost (backup)", f"{int(xgb_pred):,}", delta=f"{int(pred - xgb_pred):,}")

            st.info(f"Confidence Range: {int(pred/3):,} — {int(pred*3):,}")

# Show feature summary
st.markdown("---")
st.subheader("Input Feature Summary")
feat_df = pd.DataFrame([features]).T.rename(columns={0: "value"})
st.table(feat_df)

st.markdown("---")
st.caption("Model uses 19 tabular features + 384-dim sentence embedding for title.")
