# submit.py
import os
from kaggle.api.kaggle_api_extended import KaggleApi

def submit_to_kaggle(
    submission_path='submission.parquet', 
    competition_name='name-that-llm', 
    message='Automated submission via DeBERTa pipeline'
):
    print("--- Initializing Kaggle API ---")
    
    # 1. Verify file exists
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"❌ Error: Submission file not found at {submission_path}. Run train.py first!")
    
    # 2. Authenticate with Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
        print("✅ Kaggle API authentication successful.")
    except Exception as e:
        raise ConnectionError(
            f"❌ Authentication failed! Ensure your kaggle.json is placed correctly at ~/.kaggle/kaggle.json.\nDetails: {e}"
        )
    
    # 3. Submit the file
    print(f"🚀 Uploading {submission_path} to Kaggle competition: '{competition_name}'...")
    try:
        api.competition_submit(
            file_name=submission_path,
            message=message,
            competition=competition_name
        )
        print("🎉 Successfully submitted to Kaggle! Check your competition submissions page for the live score.")
    except Exception as e:
        print(f"❌ Submission failed: {e}")

if __name__ == "__main__":
    submit_to_kaggle(
        submission_path='submission.parquet',
        competition_name='name-that-llm',
        message='DeBERTa-v3 baseline submission'
    )
