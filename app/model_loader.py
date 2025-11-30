import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# -----------------------------
# Multimodal PyTorch Model
# -----------------------------
class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, text_embedding_dim=384):
        super().__init__()
        self.tab_fc = nn.Sequential(
            nn.Linear(tabular_input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.text_fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        )
        self.combined_fc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_tabular, x_text):
        tab_out = self.tab_fc(x_tabular)
        text_out = self.text_fc(x_text)
        combined = torch.cat([tab_out, text_out], dim=1)
        return self.combined_fc(combined).squeeze()


# -----------------------------
# Model Loader
# -----------------------------
class ModelLoader:
    def __init__(self, pytorch_model_path, xgboost_model_path=None):
        self.device = torch.device("cpu")
        self.pytorch_model_path = pytorch_model_path
        self.xgboost_model_path = xgboost_model_path
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pytorch_model = self._load_pytorch_model()
        self.xgboost_model = self._load_xgboost_model() if xgboost_model_path else None

    # -------------------------
    # Load PyTorch Model
    # -------------------------
    def _load_pytorch_model(self):
        checkpoint = torch.load(self.pytorch_model_path, map_location=self.device)
        config = checkpoint['model_config']
        model = MultimodalModel(
            tabular_input_dim=config['tabular_input_dim'],
            text_embedding_dim=config['text_embedding_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model

    # -------------------------
    # Load XGBoost Model
    # -------------------------
    def _load_xgboost_model(self):
        with open(self.xgboost_model_path, "rb") as f:
            return pickle.load(f)

    # -------------------------
    # Predict with PyTorch
    # -------------------------
    def predict_pytorch(self, features_dict, title_text):
        # Order must match training tabular_cols
        feature_list = [
            features_dict['title_length'],
            features_dict['title_word_count'],
            features_dict['uppercase_words'],
            features_dict['num_emojis'],
            features_dict['has_emoji'],
            features_dict['contains_numbers_or_emojis'],
            features_dict['has_question'],
            features_dict['is_clickbait'],
            features_dict['sentiment_polarity'],
            features_dict['sentiment_subjectivity'],
            features_dict['top50_pca1'],
            features_dict['top50_pca2'],
            features_dict['top50_pca3'],
            features_dict['is_published_weekend'],
            features_dict['category_id'],
            features_dict['comments_disabled'],
            features_dict['ratings_disabled'],
            features_dict['days_until_trending']
        ]

        X_tab = torch.tensor([feature_list], dtype=torch.float32).to(self.device)
        X_text = torch.tensor(self.sentence_model.encode([title_text], convert_to_numpy=True),
                              dtype=torch.float32).to(self.device)

        with torch.no_grad():
            log_views = self.pytorch_model(X_tab, X_text).cpu().item()
            return np.expm1(log_views)

    # -------------------------
    # Predict with XGBoost
    # -------------------------
    def predict_xgboost(self, features_dict):
        if self.xgboost_model is None:
            return None

        feature_list = [
            features_dict['title_length'],
            features_dict['title_word_count'],
            features_dict['uppercase_words'],
            features_dict['num_emojis'],
            features_dict['has_emoji'],
            features_dict['contains_numbers_or_emojis'],
            features_dict['has_question'],
            features_dict['is_clickbait'],
            features_dict['sentiment_polarity'],
            features_dict['sentiment_subjectivity'],
            features_dict['top50_pca1'],
            features_dict['top50_pca2'],
            features_dict['top50_pca3'],
            features_dict['is_published_weekend'],
            features_dict['category_id'],
            features_dict['comments_disabled'],
            features_dict['ratings_disabled'],
            features_dict['days_until_trending']
        ]

        X = np.array([feature_list])
        log_views = self.xgboost_model.predict(X)[0]
        return np.expm1(log_views)
