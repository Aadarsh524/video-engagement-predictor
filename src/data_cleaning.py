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
        """
        Initialize cleaner with raw data path
        
        Args:
            raw_data_path (str): Path to raw CSV file
        """
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
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # Log missing value counts
        missing_counts = self.df.isnull().sum()
        self.cleaning_stats['missing_before'] = missing_counts.to_dict()
        
        if missing_counts.sum() > 0:
            logger.info(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
            
            # Strategy 1: Drop rows with missing critical numeric fields
            critical_cols = ['views', 'likes', 'dislikes', 'comment_count']
            before_count = len(self.df)
            
            for col in critical_cols:
                if col in self.df.columns:
                    self.df = self.df.dropna(subset=[col])
            
            dropped = before_count - len(self.df)
            logger.info(f"Dropped {dropped} rows with missing critical values")
            self.cleaning_stats['rows_dropped_missing'] = dropped
            
            # Strategy 2: Fill text fields with empty string
            text_cols = ['title', 'channel_title', 'tags', 'description', 'category_name']
            for col in text_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('')
        else:
            logger.info("No missing values found!")
            self.cleaning_stats['rows_dropped_missing'] = 0
        
        return self
    
    def remove_duplicates(self):
        """Remove duplicate video entries"""
        logger.info("Removing duplicates...")
        before_count = len(self.df)
        
        # Remove duplicates based on video_id if available
        if 'video_id' in self.df.columns:
            self.df = self.df.drop_duplicates(subset=['video_id'], keep='first')
            logger.info(f"Removed duplicates based on video_id")
        else:
            # Fall back to removing exact duplicate rows
            self.df = self.df.drop_duplicates()
            logger.info(f"Removed exact duplicate rows")
        
        after_count = len(self.df)
        duplicates_removed = before_count - after_count
        logger.info(f"Removed {duplicates_removed} duplicate rows")
        self.cleaning_stats['duplicates_removed'] = duplicates_removed
        
        return self
    
    def convert_data_types(self):
        """Convert columns to appropriate data types"""
        logger.info("Converting data types...")
        
        # Convert numeric columns to proper types
        numeric_cols = ['views', 'likes', 'dislikes', 'comment_count', 'category_id']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                logger.info(f"Converted {col} to numeric")
        
        # Convert datetime columns
        date_cols = ['publish_date', 'trending_date', 'video_publish_data']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to datetime: {e}")
        
        # Drop rows with failed conversions in critical fields
        before_count = len(self.df)
        self.df = self.df.dropna(subset=['views', 'likes'])
        after_count = len(self.df)
        
        if before_count != after_count:
            logger.info(f"Dropped {before_count - after_count} rows with invalid data types")
        
        return self
    
    def handle_outliers(self, method='cap', threshold=1.5):
        """
        Handle outliers in numerical columns
        """
        logger.info(f"Handling outliers using {method} method (threshold={threshold})...")
        
        # Focus on views as main metric
        numeric_cols = ['views', 'likes', 'dislikes', 'comment_count']
        outlier_stats = {}
        
        for col in numeric_cols:
            if col not in self.df.columns:
                continue
                
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers_count = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            
            if method == 'cap':
                # Cap values at boundaries (keeps data, reduces extreme impact)
                self.df[col] = self.df[col].clip(lower=max(0, lower_bound), upper=upper_bound)
                outlier_stats[col] = f"{outliers_count} capped"
            elif method == 'remove':
                # Remove rows with outliers
                before = len(self.df)
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
                removed = before - len(self.df)
                outlier_stats[col] = f"{removed} removed"
        
        if outlier_stats:
            logger.info(f"Outliers handled: {outlier_stats}")
            self.cleaning_stats['outliers'] = outlier_stats
        
        return self
    

    
    def add_category_mapping(self):
        """Add category names if not already present"""
        if 'category_name' not in self.df.columns and 'category_id' in self.df.columns:
            logger.info("Adding category name mapping...")
            
            video_category = {
                1: 'Film & Animation', 2: 'Autos & Vehicles', 10: 'Music',
                15: 'Pets & Animals', 17: 'Sports', 20: 'Gaming',
                22: 'People & Blogs', 23: 'Comedy', 24: 'Entertainment',
                25: 'News & Politics', 26: 'Howto & Style', 27: 'Education',
                28: 'Science & Technology', 29: 'Nonprofits & Activism',
                43: 'Shows', 44: 'Trailers'
            }
            
            self.df['category_name'] = self.df['category_id'].map(video_category)
            logger.info("Category names added")
        
        return self

    
    def save_cleaned_data(self, output_path):
        """Save cleaned data to CSV"""
        logger.info(f"Saving cleaned data to {output_path}")
        self.df.to_csv(output_path, index=False)
        self.cleaning_stats['final_rows'] = len(self.df)
        self.cleaning_stats['final_columns'] = len(self.df.columns)
        logger.info(f"Saved {len(self.df)} rows with {len(self.df.columns)} columns")
        return self
    
    def get_cleaning_report(self):
        """Generate a comprehensive cleaning report"""
        report = f"""
          DATA CLEANING REPORT                           

📊 DATA DIMENSIONS:
   Initial:  {self.cleaning_stats.get('initial_rows', 'N/A')} rows × {self.cleaning_stats.get('initial_columns', 'N/A')} columns
   Final:    {self.cleaning_stats.get('final_rows', 'N/A')} rows × {self.cleaning_stats.get('final_columns', 'N/A')} columns
   
🔧 CLEANING OPERATIONS:
   Duplicates removed:     {self.cleaning_stats.get('duplicates_removed', 0)}
   Missing values handled: {self.cleaning_stats.get('rows_dropped_missing', 0)} rows dropped
   Invalid rows removed:   {self.cleaning_stats.get('invalid_rows_removed', 0)}
   
✨ FEATURES CREATED:
   {len(self.cleaning_stats.get('features_created', []))} new features:
   {', '.join(self.cleaning_stats.get('features_created', []))}

📈 DATA QUALITY:
   Missing values: {self.df.isnull().sum().sum()} total
   Data retention: {(self.cleaning_stats.get('final_rows', 0) / self.cleaning_stats.get('initial_rows', 1)) * 100:.1f}%

"""
        return report


def main():
    """Main pipeline execution"""
    
    print("🚀 Starting YouTube Data Cleaning Pipeline...\n")
    
    # Initialize cleaner
    cleaner = YouTubeDataCleaner('../data/raw/youtube.csv')
    
    # Run complete cleaning pipeline
    cleaner.load_data() \
           .handle_missing_values() \
           .remove_duplicates() \
           .convert_data_types() \
           .remove_invalid_rows() \
           .add_category_mapping() \
           .handle_outliers(method='cap', threshold=3) \
           .create_engagement_features() \
           .save_cleaned_data('../data/processed/youtube_cleaned.csv')
    
    # Print comprehensive report
    print(cleaner.get_cleaning_report())
    
    # Print sample of cleaned data
    print("\n📋 SAMPLE OF CLEANED DATA:")
    print(cleaner.df.head())
    
    print("\n✅ Cleaning complete! Check ../data/processed/youtube_cleaned.csv")


if __name__ == "__main__":
    main()