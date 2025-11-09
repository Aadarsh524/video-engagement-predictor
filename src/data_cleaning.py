"""
Data Cleaning Pipeline for YouTube Trending Video Statistics
Customized for your specific dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class YouTubeDataCleaner:
    """
    Cleans and preprocesses YouTube trending video data
    """
    
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.df = None
        self.cleaning_stats = {}
        
    def load_data(self):
        """Load raw data from CSV"""
        logger.info(f"Loading data from {self.raw_data_path}")
        self.df = pd.read_csv(self.raw_data_path)
        self.cleaning_stats['initial_rows'] = len(self.df)
        self.cleaning_stats['initial_columns'] = len(self.df.columns)
        logger.info(f"Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self
    
    def handle_missing_values(self):
        logger.info("Handling missing values...")
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        missing_counts = self.df.isnull().sum()
        self.cleaning_stats['missing_before'] = missing_counts.to_dict()
        
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            critical_cols = ['views', 'likes', 'dislikes', 'comment_count']
            before_count = len(self.df)
            
            for col in critical_cols:
                if col in self.df.columns:
                    self.df = self.df.dropna(subset=[col])
            
            dropped = before_count - len(self.df)
            self.cleaning_stats['rows_dropped_missing'] = dropped
            
            text_cols = ['title', 'channel_title', 'tags', 'description', 'category_name']
            for col in text_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('')
        else:
            logger.info("No missing values found!")
            self.cleaning_stats['rows_dropped_missing'] = 0
        
        return self
    
    def remove_duplicates(self):
        logger.info("Removing duplicates...")
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        before_count = len(self.df)
        if 'video_id' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['video_id'], keep='first')
        else:
            self.df = self.df.drop_duplicates()
        
        after_count = len(self.df)
        self.cleaning_stats['duplicates_removed'] = before_count - after_count
        return self
    
    def convert_data_types(self):
        logger.info("Converting data types...")
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        numeric_cols = ['views', 'likes', 'dislikes', 'comment_count', 'category_id']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        date_cols = ['publish_date', 'trending_date']
        for col in date_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        return self
    
    def handle_outliers(self, method='cap', threshold=1.5):
        logger.info(f"Handling outliers using {method} (threshold={threshold})...")
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        numeric_cols = ['views', 'likes', 'dislikes', 'comment_count']
        outlier_stats = {}
        
        for col in numeric_cols:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            
            if method == 'cap':
                self.df[col] = self.df[col].clip(lower=max(0, lower), upper=upper)
                outlier_stats[col] = f"{count} capped"
            elif method == 'remove':
                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                outlier_stats[col] = f"{count} removed"
        
        self.cleaning_stats['outliers'] = outlier_stats
        return self
    
    def add_category_mapping(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if 'category_name' not in self.df.columns and 'category_id' in self.df.columns:
            video_category = {
                1: 'Film & Animation', 2: 'Autos & Vehicles', 10: 'Music',
                15: 'Pets & Animals', 17: 'Sports', 20: 'Gaming',
                22: 'People & Blogs', 23: 'Comedy', 24: 'Entertainment',
                25: 'News & Politics', 26: 'Howto & Style', 27: 'Education',
                28: 'Science & Technology', 29: 'Nonprofits & Activism',
                43: 'Shows', 44: 'Trailers'
            }
            self.df['category_name'] = self.df['category_id'].map(video_category)
        return self
    
    def save_cleaned_data(self, output_path):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.df.to_csv(output_path, index=False)
        self.cleaning_stats['final_rows'] = len(self.df)
        self.cleaning_stats['final_columns'] = len(self.df.columns)
        return self
    
    def get_cleaning_report(self):
        if self.df is None:
            return "⚠️ No data loaded yet. Run load_data() and cleaning steps first."
        
        report = f"""
        🧹 DATA CLEANING REPORT

📊 DATA DIMENSIONS:
   Initial: {self.cleaning_stats.get('initial_rows', 'N/A')} rows × {self.cleaning_stats.get('initial_columns', 'N/A')} columns
   Final:   {self.cleaning_stats.get('final_rows', 'N/A')} rows × {self.cleaning_stats.get('final_columns', 'N/A')} columns

🔧 CLEANING OPERATIONS:
   Duplicates removed:     {self.cleaning_stats.get('duplicates_removed', 0)}
   Missing values handled: {self.cleaning_stats.get('rows_dropped_missing', 0)}
   Outliers handled:       {self.cleaning_stats.get('outliers', {})}

📈 DATA QUALITY:
   Remaining missing values: {self.df.isnull().sum().sum()} total
   Data retention: {(self.cleaning_stats.get('final_rows', 0) / self.cleaning_stats.get('initial_rows', 1)) * 100:.1f}%

"""
        return report
