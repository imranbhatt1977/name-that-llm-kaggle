# test_submission.py
import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def validate_submission(submission_path='submission.parquet', test_path='test.csv', train_path='train.csv'):
    print(f"--- Validating Submission File: {submission_path} ---")
    
    # 1. Check if submission file exists
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"❌ Error: Submission file not found at {submission_path}. Run train.py first!")
        
    # 2. Load files
    sub_df = pd.read_parquet(submission_path)
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    
    errors = []
    
    # 3. Check Column Names
    expected_columns = ['id', 'model']
    if list(sub_df.columns) != expected_columns:
        errors.append(f"❌ Columns mismatch! Expected {expected_columns}, but got {list(sub_df.columns)}.")
    else:
        print("✅ Column names match expected format ('id', 'model').")
        
    # 4. Check Row Counts
    if len(sub_df) != len(test_df):
        errors.append(f"❌ Row count mismatch! Submission has {len(sub_df)} rows, but test set has {len(test_df)} rows.")
    else:
        print(f"✅ Row count is correct: {len(sub_df)} rows.")
        
    # 5. Check for Missing (NaN) Values
    if sub_df.isnull().any().any():
        nan_counts = sub_df.isnull().sum().to_dict()
        errors.append(f"❌ Missing values detected in submission: {nan_counts}")
    else:
        print("✅ No missing (NaN) values found.")
        
    # 6. Check 'id' alignment
    if 'id' in sub_df.columns and 'id' in test_df.columns:
        if not np.array_equal(sub_df['id'].values, test_df['id'].values):
            errors.append("❌ ID mismatch! The 'id' column order or values do not align with the test set.")
        else:
            print("✅ All test IDs match and are in the correct order.")
            
    # 7. Check that predicted classes match valid training labels
    if 'model' in sub_df.columns:
        valid_classes = set(train_df['model'].unique())
        predicted_classes = set(sub_df['model'].unique())
        
        unknown_classes = predicted_classes - valid_classes
        if unknown_classes:
            errors.append(f"❌ Invalid model labels predicted that were never in train set: {unknown_classes}")
        else:
            print(f"✅ All predicted labels belong to the 18 valid training model classes.")
            
        print("\n📊 Predicted Class Distribution Summary:")
        print(sub_df['model'].value_counts())

    # --- FINAL REPORT ---
    print("\n" + "="*40)
    if errors:
        print("🚨 SUBMISSION FAILED VALIDATION:")
        for err in errors:
            print(err)
        raise ValueError("Fix the errors above before submitting to Kaggle!")
    else:
        print("🎉 SUCCESS! Submission file is fully valid and ready for Kaggle upload.")
    print("="*40)

if __name__ == "__main__":
    validate_submission()
