📺 Video Engagement Predictor

Predicting YouTube video engagement using metadata + NLP features + machine learning + XGBoost + deep learning.

Live Demo → https://video-engagement-predictor-3srm9eebk76ygzhye8epdw.streamlit.app/

🚀 Project Overview

This project builds a complete end-to-end system that predicts video engagement (views) using:

✔ Metadata
✔ Text features extracted from video titles
✔ Emoji features
✔ Sentiment analysis
✔ Top-50 keyword PCA embeddings
✔ Multimodal ML models (regression + XGBoost + RandomForest)
✔ Full deployment via Streamlit & Docker

The goal: Understand what makes videos go viral and create an ML system that forecasts engagement before upload.


📚 Project Structure
video-engagement-predictor/
│── data/
│   ├── raw/
│   └── processed/
│── models/
│── notebooks/
│── src/
│   ├── data_cleaning.py
│   ├── features.py
│   ├── evaluation.py
│   └── pipeline.py
│── streamlit_app/
│── tests/
│── Dockerfile
│── requirements.txt
└── README.md


🧹 Data Cleaning Summary
📊 Dataset Dimensions
Stage	Rows	Columns
Initial	161,470	18
Final	55,886	18

🔧 Cleaning Operations

Duplicates removed: 105,584
views: 3,270 capped
likes: 4,272 capped
dislikes: 4,117 capped
comment_count: 3,839 capped

📈 Data Quality
Remaining missing values: 0
Retained data: 34.6%


📈 Views Percentile Analysis
Percentile	Views
50th	97,096
75th	288,748
90th	723,184
95th	1,080,586
99th	1,080,586
99.9th	1,080,586

Viral videos are extremely skewed, with 1M+ views dominating the top percentiles.

🔥 Top Categories for Viral Videos
Category	Count
Entertainment	1749
Music	927
Sports	711
Comedy	666
People & Blogs	482
Howto & Style	313
Film & Animation	310
News & Politics	263
Science & Technology	204
Gaming	204


🧠 Feature Engineering

📝 Title-based Features
title length
uppercase words
title word count
has question mark
contains numbers / emoji
clickbait phrase flag

😀 Emoji Features
num_emojis
has_emoji
emojis_list

❤️ Sentiment Features
Using TextBlob:
sentiment polarity
sentiment subjectivity
🔤 Top-50 Word Features (NLP)
Binary presence vector
PCA reduced embeddings:
top50_pca1
top50_pca2
top50_pca3

🕒 Time Features
published_day_of_week
is_published_weekend
days_until_trending


Model Performance Summary

🏆 Final Ranking (Best → Worst)
Rank	Model	RMSE	MAE	R²
🥇 1	Advanced Multimodal	1.2759	1.0005	0.4220
🥈 2	Tuned XGBoost	1.3563	1.0608	0.3468
🥉 3	Random Forest	1.3759	1.0799	0.3278
   4	Linear Regression	1.5593	1.2762	0.1367


🖥️ Streamlit Demo

Live web app (free hosting):

👉 https://video-engagement-predictor-3srm9eebk76ygzhye8epdw.streamlit.app/

Features:
Enter video metadata
NLP processing of title
Predict expected views
Show feature contributions
Interactive visualizations


🐳 Docker Image
This project can be run inside a Docker container.
Dockerfile
Contains:
    Python 3.10 base image
    All dependencies from requirements.txt
    Streamlit exposure on port 8501=
    Entrypoint for running the app
Build
docker build -t video-engagement-predictor .

Run
docker run -p 8501:8501 video-engagement-predictor


Streamlit app becomes available at:
👉 http://localhost:8501

📦 Installation (Local Setup)
1️⃣ Clone Repository
git clone https://github.com/<your-username>/video-engagement-predictor
cd video-engagement-predictor

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Launch Jupyter
jupyter notebook

4️⃣ Run Streamlit App
streamlit run streamlit_app/app.py



👨‍💻 Author
Aadarsh Ghimire
Machine Learning | Data Engineering | AI Systems
Sydney, Australia