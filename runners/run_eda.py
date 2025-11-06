import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from eda_pipeline import EdaPipeline
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './data')))

    raw_data_path = './data/raw/youtube.csv'
    EdaPipeline(raw_data_path, output_prefix="raw").run_pipeline()

    clean_data_path = './data/processed/youtube_cleaned.csv'
    EdaPipeline(clean_data_path, output_prefix="cleaned").run_pipeline()

    logging.info("✅ Both raw and cleaned EDA reports generated.")


