# app.py — very first lines, absolutely nothing above
import torch

# Now import everything else
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from model_loader import ModelLoader
from utils import extract_features_from_title, CATEGORIES, DAYS, compute_top50_features
device = torch.device("cpu")


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="YouTube Views Predictor",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# LOAD MODELS (CACHED)
# -----------------------------
@st.cache_resource
def load_models():
    loader = ModelLoader(
        pytorch_model_path='models/multimodal_pytorch.pth',
        xgboost_model_path='models/xgboost_tuned.pkl'
    )
    return loader

model_loader = load_models()

# -----------------------------
# SIDEBAR INFO
# -----------------------------
st.sidebar.title("📊 Model Information")
st.sidebar.markdown("""
**Best Model:** PyTorch Multimodal  
**R² Score:** 0.2746 (27.5% variance)  
**RMSE:** 1.3021 (log scale)

**Features Used:**
- Video metadata (14 features)
- Top-50 PCA embeddings (3 features)
- Title text embeddings (384-dim)
""")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Enter realistic video details for better predictions!")

# -----------------------------
# TOP-50 WORDS FOR PCA
# -----------------------------
TOP_50_WORDS = [
    '10', '2017', '2018', 'audio', 'avec', 'best', 'black', 'challenge', 'christmas',
    'clip', 'clip officiel', 'day', 'en', 'ep', 'episode', 'et', 'feat', 'food', 'ft',
    'game', 'game highlights', 'goals', 'hd', 'highlights', 'house', 'la', 'le',
    'life', 'live', 'love', 'music', 'nba', 'new', 'news', 'official',
    'official trailer', 'official video', 'officiel', 'real', 'season', 'songs',
    'star', 'time', 'trailer', 'trump', 'tv', 'video', 'vs', 'world', 'ðÿ'
]

# -----------------------------
# MAIN APP
# -----------------------------
st.title("📺 YouTube Video Views Predictor")
st.markdown("""
Predict potential views for your YouTube video using AI! This model uses **multimodal deep learning** 
combining metadata and text analysis.
""")
st.markdown("---")

# -----------------------------
# INPUT SECTION
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Video Details")
    title = st.text_input("Video Title", value="10 Amazing Python Tips for Beginners 🐍")
    category = st.selectbox("Category", options=list(CATEGORIES.keys()), index=list(CATEGORIES.keys()).index('Education'))
    
    col_a, col_b = st.columns(2)
    with col_a:
        publish_day = st.selectbox("Publication Day", options=list(DAYS.keys()), index=1)
    with col_b:
        hour_of_trending = st.slider("Hour of Trending (0-23)", 0, 23, 14)
    
    days_until_trending = st.number_input("Days Until Trending", min_value=0, max_value=30, value=2)
    
    col_c, col_d = st.columns(2)
    with col_c:
        comments_disabled = st.checkbox("Comments Disabled", value=False)
        is_english = st.checkbox("English Video", value=True)
    with col_d:
        ratings_disabled = st.checkbox("Ratings Disabled", value=False)

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
df_title = pd.DataFrame({'title_cleaned': [title.lower()]})
df_title = compute_top50_features(df_title, cleaned_col='title_cleaned', top_words=TOP_50_WORDS)
title_features = extract_features_from_title(title)

features = {
    **title_features,
    'category_id': CATEGORIES[category],
    'published_day_of_week_num': DAYS[publish_day],
    'hour_of_trending': hour_of_trending,
    'days_until_trending': days_until_trending,
    'comments_disabled': int(comments_disabled),
    'ratings_disabled': int(ratings_disabled),
    'is_english': int(is_english),
    'top50_pca1': df_title['top50_pca1'].iloc[0],
    'top50_pca2': df_title['top50_pca2'].iloc[0],
    'top50_pca3': df_title['top50_pca3'].iloc[0],
    'is_title_english': df_title['is_title_english'].iloc[0],
    'is_published_weekend': int(DAYS[publish_day] >= 5),
    'is_trending_weekend': int(hour_of_trending >= 5),
}

# -----------------------------
# PREDICTION
# -----------------------------
with col2:
    st.subheader("🎯 Predicted Views")
    if st.button("🚀 Predict Views"):
        with st.spinner("Analyzing..."):
            pytorch_pred = model_loader.predict_pytorch(features, title)
           
           
            
            xgb_pred = model_loader.predict_xgboost(features) if model_loader.xgboost_model else None
            
            st.metric("Predicted Views (Multimodal)", f"{int(pytorch_pred):,}")
            
            if xgb_pred:
                st.metric("XGBoost Prediction (Backup)", f"{int(xgb_pred):,}", delta=f"{int(pytorch_pred - xgb_pred):,} vs Multimodal")
            
            lower_bound = pytorch_pred / 3
            upper_bound = pytorch_pred * 3
            st.info(f"**Confidence Range:** {int(lower_bound):,} - {int(upper_bound):,} views")
            
            if pytorch_pred < 10000:
                st.warning("📊 Low virality - Consider optimizing title/timing")
            elif pytorch_pred < 100000:
                st.success("📈 Moderate performance expected")
            else:
                st.balloons()
                st.success("🔥 High viral potential!")

# -----------------------------
# FEATURE ANALYSIS
# -----------------------------
st.markdown("---")
st.subheader("🔍 Feature Analysis")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Title Length", f"{features['title_length']} chars")
    st.metric("Uppercase Words", features['uppercase_words'])
with col2:
    st.metric("Emojis Count", features['num_emojis'])
    st.metric("Day of Week", publish_day)
with col3:
    st.metric("Hour of Trending", f"{hour_of_trending}:00")
    st.metric("Days to Trend", days_until_trending)

# -----------------------------
# VISUALIZATION
# -----------------------------
st.markdown("---")
st.subheader("📊 Model Comparison")
comparison_data = {
    'Model': ['Baseline', 'Linear Reg', 'XGBoost', 'Random Forest', 'PyTorch\nMultimodal'],
    'R² Score': [0.0000, 0.0651, 0.2448, 0.2258, 0.2746],
    'RMSE': [1.5281, 1.4781, 1.3285, 1.3451, 1.3021]
}
fig = go.Figure()
fig.add_trace(go.Bar(x=comparison_data['Model'], y=comparison_data['R² Score'], name='R² Score', marker_color='lightblue'))
fig.update_layout(title="Model Performance Comparison", xaxis_title="Model", yaxis_title="R² Score", height=400)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("""
**🎓 Project:** YouTube Video Views Prediction using Multimodal Deep Learning  
**📊 Dataset:** 26,678 trending YouTube videos  
**🤖 Models:** PyTorch (Tabular + Text), XGBoost, Random Forest
""")
