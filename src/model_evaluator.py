import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error  # ✅ Fixed: use mean_squared_error, not root_mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

day_map = {
    'monday': 0,
    'tuesday': 1,
    'wednesday': 2,
    'thursday': 3,
    'friday': 4,
    'saturday': 5,
    'sunday': 6
}

features = [
    
    # --- Title / NLP features ---
    'title_length',
    'title_word_count',
    'uppercase_words',
    'num_emojis',
    'has_emoji',
    'contains_numbers_or_emojis',
    'has_question',
    'is_clickbait',
    'sentiment_polarity',
    'sentiment_subjectivity',

    # ---  content ---
    'top50_pca1',
    'top50_pca2',
    'top50_pca3',

    # --- Time Features ---
    'is_published_weekend',

    # --- Metadata ---
    'category_id',
    'comments_disabled',
    'ratings_disabled',


     'days_until_trending',   
]

class ModelEvaluator:  
    def __init__(self, complete_data_path):
        self.complete_data_path = complete_data_path
        self.models_stats = {}  
        self.df = None  
        self.X_train = None 
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load complete featured data from CSV"""
        logger.info(f"Loading data from {self.complete_data_path}")
        self.df = pd.read_csv(self.complete_data_path)
        logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        self.df['published_day_of_week_num'] = self.df['published_day_of_week'].str.lower().map(day_map)
        return self
    
    def prepare_train_test_split(self):  
        """Prepare train/test split with new features"""
        X = self.df[features]
        y = np.log1p(self.df['views'])
    
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
        logger.info(f"Train set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        return self

    
    def linear_regression(self):
        """Train and evaluate Linear Regression model"""
        logger.info("Training Linear Regression...")
        
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        y_pred = lr.predict(self.X_test)
        
        # ✅ Fixed: use mean_squared_error, not mean_absolute_error for RMSE
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"\nLinear Regression Model Performance (LOG SCALE):")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        
      
        # Use coefficients instead
        print("\n" + "="*60)
        print("Linear Regression COEFFICIENTS")
        print("="*60)
        
        lr_coefficients = list(zip(features, lr.coef_))
        lr_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for i, (feature, coef) in enumerate(lr_coefficients, 1):
            bar = '█' * int(abs(coef) * 10)  # Scale for visualization
            print(f"{i:2d}. {feature:<30} {coef:>8.4f}  {bar}")
        
        self.models_stats['linear_regression'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'coefficients': lr_coefficients,
            'predictions': y_pred  # ✅ Added: store predictions for plotting
        }
        
        return self

    def xgboost(self):
        """Train and evaluate default XGBoost model"""
        logger.info("Training XGBoost (default parameters)...")
        
        xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
        xgb_model.fit(self.X_train, self.y_train)

        y_pred = xgb_model.predict(self.X_test)
        
        
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        print(f"\nXGBoost Model Performance (LOG SCALE):")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        
        print("\n" + "="*60)
        print("XGBoost FEATURE IMPORTANCE")  # ✅ Fixed: label
        print("="*60)
        
        xgb_importance = xgb_model.feature_importances_
        xgb_feature_importance = sorted(
            zip(features, xgb_importance), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for i, (feature, importance) in enumerate(xgb_feature_importance, 1):
            bar = '█' * int(importance * 100)
            print(f"{i:2d}. {feature:<30} {importance:>7.4f}  {bar}")
         
        self.models_stats['xgboost_default'] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'feature_importance': xgb_feature_importance,
            'predictions': y_pred,
            'model': xgb_model  
        }
        
        return self

    def tuned_xgboost(self):
        """Train and evaluate hyperparameter-tuned XGBoost"""
        logger.info("Starting XGBoost hyperparameter tuning...")
        
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9, 12],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.5],
        }
        
        xgb_base = xgb.XGBRegressor(
            random_state=42, 
            verbosity=0,
            n_jobs=-1
        )
        
        random_search = RandomizedSearchCV(
            estimator=xgb_base,
            param_distributions=param_distributions,
            n_iter=50,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42,
            verbose=2,
            n_jobs=-1
        )
        
      
        random_search.fit(self.X_train, self.y_train)

        print("\n✅ Random Search fitting complete!")
        
        best_params = random_search.best_params_
        print("\n" + "="*60)
        print("BEST HYPERPARAMETERS FOUND")
        print("="*60)
        for param, value in best_params.items():
            print(f"  {param:<20}: {value}")
        
        best_xgb = random_search.best_estimator_
        y_pred_tuned = best_xgb.predict(self.X_test)
        
        
        rmse_tuned = np.sqrt(mean_squared_error(self.y_test, y_pred_tuned))
        mae_tuned = mean_absolute_error(self.y_test, y_pred_tuned)
        r2_tuned = r2_score(self.y_test, y_pred_tuned)
        
        print("\n" + "="*60)
        print("TUNED XGBOOST PERFORMANCE")
        print("="*60)
        print(f"RMSE: {rmse_tuned:.4f}")
        print(f"MAE:  {mae_tuned:.4f}")
        print(f"R²:   {r2_tuned:.4f}")
        
        print("\n" + "="*60)
        print("Tuned XGBOOST FEATURE IMPORTANCE")
        print("="*60)
        
        importance_scores = best_xgb.feature_importances_
        feature_importance = list(zip(features, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop Features (sorted by importance):")
        for i, (feature, importance) in enumerate(feature_importance, 1):
            bar = '█' * int(importance * 100)
            print(f"{i:2d}. {feature:<30} {importance:>6.4f} {bar}")

        self.models_stats['xgboost_tuned'] = {
            'rmse': rmse_tuned,
            'mae': mae_tuned,
            'r2': r2_tuned,
            'feature_importance': feature_importance,
            'predictions': y_pred_tuned,
            'model': best_xgb,
            'best_params': best_params
        }
        
        return self

    def randomforest(self):
        """Train and evaluate Random Forest model"""
        logger.info("Training Random Forest...")
        
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        rf_model.fit(self.X_train, self.y_train)
        logger.info("✅ Random Forest trained!")
        
        y_pred_rf = rf_model.predict(self.X_test)

       
        rmse_rf = np.sqrt(mean_squared_error(self.y_test, y_pred_rf))
        mae_rf = mean_absolute_error(self.y_test, y_pred_rf)
        r2_rf = r2_score(self.y_test, y_pred_rf)
        
        print("\n" + "="*60)
        print("RANDOM FOREST PERFORMANCE")
        print("="*60)
        print(f"RMSE: {rmse_rf:.4f}")
        print(f"MAE:  {mae_rf:.4f}")
        print(f"R²:   {r2_rf:.4f}")

        print("\n" + "="*60)
        print("RANDOM FOREST FEATURE IMPORTANCE")
        print("="*60)
        
        rf_importance = rf_model.feature_importances_
        rf_feature_importance = sorted(
            zip(features, rf_importance), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for i, (feature, importance) in enumerate(rf_feature_importance, 1):
            bar = '█' * int(importance * 50)
            print(f"{i:2d}. {feature:<30} {importance:>7.4f}  {bar}")
        
        self.models_stats['random_forest'] = {
            'rmse': rmse_rf,
            'mae': mae_rf,
            'r2': r2_rf,
            'feature_importance': rf_feature_importance,
            'predictions': y_pred_rf,
            'model': rf_model
        }
        
        return self

    def get_models_report(self):
        """Generate comprehensive model comparison report"""
        if self.df is None:
            return "⚠️ No data loaded yet. Run load_data() first."
        
        if not self.models_stats:
            return "⚠️ No models trained yet. Run model training methods first."
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                 MODEL COMPARISON REPORT                      ║
╚══════════════════════════════════════════════════════════════╝

Dataset: {len(self.df)} videos
Features: {len(features)}
Train/Test Split: {len(self.X_train)}/{len(self.X_test)}

"""
        
        # ✅ Create comparison table
        print("\n" + "="*80)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*80)
        print(f"{'Model':<25} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
        print("-"*80)
        
        # Sort by R² (descending)
        sorted_models = sorted(
            self.models_stats.items(),
            key=lambda x: x[1]['r2'],
            reverse=True
        )
        
        for model_name, stats in sorted_models:
            print(f"{model_name:<25} {stats['rmse']:<12.4f} {stats['mae']:<12.4f} {stats['r2']:<12.4f}")
        
        print("="*80)
        
        # Find best model
        best_model_name = sorted_models[0][0]
        best_stats = sorted_models[0][1]
        
        report += f"""
 BEST MODEL: {best_model_name}
   • RMSE: {best_stats['rmse']:.4f}
   • MAE:  {best_stats['mae']:.4f}
   • R²:   {best_stats['r2']:.4f} ({best_stats['r2']*100:.2f}% variance explained)

"""
        
        return report
    

    def plot_predictions(self, model_name):

        y_true = self.y_test
        y_pred = self.models_stats[model_name]['predictions']
    
        plt.figure(figsize=(7,7))
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()])
        plt.xlabel("Actual (log views)")
        plt.ylabel("Predicted (log views)")
        plt.title(f"{model_name} — Prediction vs Actual")
        plt.show()

    
    def save_best_model(self, output_path='../models/best_model.txt'):
        """Save the best performing model"""
        import pickle
        import os
        
        # Find best model by R²
        best_model_name = max(self.models_stats.items(), key=lambda x: x[1]['r2'])[0]
        best_model = self.models_stats[best_model_name]['model']
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        logger.info(f"✅ Best model ({best_model_name}) saved to: {output_path}")
        return output_path

