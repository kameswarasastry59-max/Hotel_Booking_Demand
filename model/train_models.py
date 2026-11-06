"""
Hotel Booking Demand Prediction - Model Training & Evaluation Pipeline
BITS Pilani WILP M.Tech (AIML) - Machine Learning Assignment 2

Trains and evaluates 5 classification models:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier (KNN)
4. Naive Bayes Classifier (GaussianNB)
5. Support Vector Machine (SVM)

Computes the 6 mandatory evaluation metrics:
- Accuracy
- AUC Score (ROC-AUC)
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)
"""

import os
import json
import time
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Set environment variable for matplotlib temporary cache
os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', '/tmp')


def preprocess_raw_dataframe(raw_df: pd.DataFrame):
    """
    Applies domain cleaning and feature engineering to raw hotel booking dataframe.
    """
    df = raw_df.copy()
    
    # 1. Domain-specific missing value imputation
    if 'country' in df.columns:
        df['country'] = df['country'].fillna('Unknown')
    if 'agent' in df.columns:
        df['agent'] = df['agent'].fillna(0).astype(int)
    if 'company' in df.columns:
        df['company'] = df['company'].fillna(0).astype(int)
    if 'children' in df.columns:
        df['children'] = df['children'].fillna(0).astype(int)

    # 2. Filter physically invalid records
    guest_sum = df['adults'] + df['children'] + df['babies']
    df = df[guest_sum > 0]
    df = df[(df['adr'] >= 0) & (df['adr'] < 5000)]

    # 3. Domain Feature Engineering
    df['total_stay_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df['total_guests'] = df['adults'] + df['children'] + df['babies']
    df['room_type_changed'] = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)
    df['is_family'] = ((df['children'] > 0) | (df['babies'] > 0)).astype(int)

    top_countries = ['PRT', 'GBR', 'FRA', 'ESP', 'DEU', 'ITA', 'IRL', 'BEL', 'BRA', 'NLD']
    df['country_group'] = df['country'].apply(lambda c: c if c in top_countries else 'Other')

    # 4. Remove target leakage columns and redundant post-booking features
    drop_leakage = [
        'reservation_status', 'reservation_status_date',
        'arrival_date_year', 'assigned_room_type', 'country'
    ]
    df = df.drop(columns=[c for c in drop_leakage if c in df.columns])

    return df


def main():
    print("=" * 70)
    print("  Hotel Booking Demand: 5-Model Training & Evaluation Pipeline")
    print("=" * 70)

    data_path = os.path.join('data', 'hotel_bookings.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading raw dataset from {data_path}...")
    raw_df = pd.read_csv(data_path)
    print(f"Loaded raw records: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns.")

    # Clean & feature engineer
    df_cleaned = preprocess_raw_dataframe(raw_df)
    print(f"Cleaned dataset shape: {df_cleaned.shape[0]:,} rows, {df_cleaned.shape[1]} columns.")

    # Create stratified sample (15,000 records) for robust training, fast prediction, and low footprint
    sample_df = df_cleaned.sample(n=15000, random_state=42).reset_index(drop=True)

    num_cols = [
        'lead_time', 'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'booking_changes', 'days_in_waiting_list', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests',
        'total_stay_nights', 'total_guests', 'room_type_changed', 'is_family'
    ]

    cat_cols = [
        'hotel', 'arrival_date_month', 'meal', 'market_segment',
        'distribution_channel', 'deposit_type', 'customer_type',
        'reserved_room_type', 'country_group'
    ]

    feature_cols = num_cols + cat_cols
    X = sample_df[feature_cols]
    y = sample_df['is_canceled']

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Training split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")
    print(f"Cancellation rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    print("Fitting preprocessor on training data...")
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    print(f"Processed feature dimensions: {X_train_proc.shape[1]} columns.")

    # Save test_data.csv for Streamlit upload and local evaluation
    # We include all input features + is_canceled target
    test_export_df = X_test.copy()
    test_export_df['is_canceled'] = y_test.values
    test_csv_path = 'test_data.csv'
    test_export_df.to_csv(test_csv_path, index=False)
    print(f"Saved {test_csv_path} with {len(test_export_df)} records.")

    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    joblib.dump(preprocessor, os.path.join('model', 'preprocessor.joblib'))
    with open(os.path.join('model', 'feature_columns.json'), 'w') as f:
        json.dump({'num_cols': num_cols, 'cat_cols': cat_cols, 'all_features': feature_cols}, f, indent=2)

    # Initialize 5 Classification Models
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            os.path.join('model', 'logistic_regression.joblib')
        ),
        'Decision Tree': (
            DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42),
            os.path.join('model', 'decision_tree.joblib')
        ),
        'KNN': (
            KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1),
            os.path.join('model', 'knn.joblib')
        ),
        'Naive Bayes': (
            GaussianNB(),
            os.path.join('model', 'naive_bayes.joblib')
        ),
        'Support Vector Machine': (
            SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
            os.path.join('model', 'svm.joblib')
        )
    }

    metrics_records = []
    trained_models = {}

    print("\nTraining and evaluating models...")
    print("-" * 70)

    for name, (model, save_path) in models.items():
        t0 = time.time()
        print(f"--> Training {name}...")
        model.fit(X_train_proc, y_train)
        elapsed = time.time() - t0

        # Predict
        y_pred = model.predict(X_test_proc)
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test_proc)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_prob = model.decision_function(X_test_proc)
        else:
            y_prob = y_pred

        # Calculate the 6 mandatory metrics
        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_prob))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        mcc = float(matthews_corrcoef(y_test, y_pred))

        # Save model
        joblib.dump(model, save_path)
        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)

        metrics_records.append({
            'ML Model Name': name,
            'Accuracy': round(acc, 4),
            'AUC': round(auc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'MCC': round(mcc, 4),
            'Training Time (s)': round(elapsed, 2),
            'Artifact Size (MB)': round(file_size_mb, 2)
        })

        print(f"    Done in {elapsed:.2f}s | Acc: {acc:.4f} | AUC: {auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | Saved: {save_path} ({file_size_mb:.2f} MB)")

    # Save summary
    results_df = pd.DataFrame(metrics_records)
    results_df.to_csv(os.path.join('model', 'comparison_table.csv'), index=False)
    with open(os.path.join('model', 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_records, f, indent=2)

    print("\n" + "=" * 70)
    print("                      ASSIGNMENT EVALUATION SUMMARY TABLE")
    print("=" * 70)
    display_cols = ['ML Model Name', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    print(results_df[display_cols].to_string(index=False))
    print("=" * 70)

    print("\nPipeline execution completed successfully!")


if __name__ == '__main__':
    main()
