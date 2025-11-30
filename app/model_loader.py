import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, text_embedding_dim=384):
        super().__init__()
        self.tab_fc = nn.Sequential(nn.Linear(tabular_input_dim, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU())
        self.text_fc = nn.Sequential(nn.Linear(text_embedding_dim, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU())
        self.combined_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                         nn.Linear(64, 1))
    def forward(self, x_tabular, x_text):
        tab_out = self.tab_fc(x_tabular)
        text_out = self.text_fc(x_text)
        combined = torch.cat([tab_out, text_out], dim=1)
        return self.combined_fc(combined).squeeze()

class ModelLoader:
    def __init__(self, pytorch_model_path, xgboost_model_path=None):
        self.device = torch.device("cpu")
        self.pytorch_model_path = pytorch_model_path
        self.xgboost_model_path = xgboost_model_path
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.pytorch_model = self._load_pytorch_model()
        self.xgboost_model = self._load_xgboost_model() if xgboost_model_path else None

    def _load_pytorch_model(self):
        # Load checkpoint fully (trusted file)
        checkpoint = torch.load(self.pytorch_model_path, map_location=self.device, weights_only=False)
    
        config = checkpoint['model_config']
        model = MultimodalModel(
            tabular_input_dim=config['tabular_input_dim'],
            text_embedding_dim=config['text_embedding_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def _load_xgboost_model(self):
        with open(self.xgboost_model_path, "rb") as f:
            return pickle.load(f)

    def predict_pytorch(self, tabular_features: dict, title_text: str):
        feature_list = [
            tabular_features['title_length'], tabular_features['uppercase_words'],
            tabular_features['sentiment_polarity'], tabular_features['sentiment_subjectivity'],
            tabular_features['category_id'], tabular_features['published_day_of_week_num'],
            tabular_features['hour_of_trending'], tabular_features['days_until_trending'],
            tabular_features['num_emojis'], tabular_features['has_emoji'],
            tabular_features['contains_numbers_or_emojis'], tabular_features['comments_disabled'],
            tabular_features['ratings_disabled'], tabular_features['is_title_english'],
            tabular_features.get('top50_pca1', 0.0), tabular_features.get('top50_pca2', 0.0),
            tabular_features.get('top50_pca3', 0.0), tabular_features.get('is_published_weekend', 0),
            tabular_features.get('is_trending_weekend', 0)
        ]
        X_tabular = torch.tensor([feature_list], dtype=torch.float32).to(self.device)
        X_text = torch.tensor(self.sentence_model.encode([title_text], convert_to_numpy=True),
                              dtype=torch.float32).to(self.device)
        with torch.no_grad():
            log_views = self.pytorch_model(X_tabular, X_text).cpu().item()
            return np.expm1(log_views)

    def predict_xgboost(self, tabular_features: dict):
        if self.xgboost_model is None: return None
        feature_list = [
            tabular_features['title_length'], tabular_features['uppercase_words'],
            tabular_features['sentiment_polarity'], tabular_features['sentiment_subjectivity'],
            tabular_features['category_id'], tabular_features['published_day_of_week_num'],
            tabular_features['hour_of_trending'], tabular_features['days_until_trending'],
            tabular_features['num_emojis'], tabular_features['has_emoji'],
            tabular_features['contains_numbers_or_emojis'], tabular_features['comments_disabled'],
            tabular_features['ratings_disabled'], tabular_features['is_title_english']
        ]
        X = np.array([feature_list])
        log_views = self.xgboost_model.predict(X)[0]
        return np.expm1(log_views)
