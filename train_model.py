# ==============================================================================
# MEDI-AI SYSTEM: MACHINE LEARNING MODEL TRAINING & SERIALIZATION SCRIPT
# Environment: Python 3.13.2 | Framework: Scikit-Learn / Pandas / Joblib
# Architecture: Random Forest Ensemble Classifier (n_estimators=100)
# ==============================================================================

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("[INFO] Initializing Medi-AI Training Pipeline...")
    print(f"[INFO] Current Working Directory: {os.getcwd()}")
    
    # 1. Dataset Ingestion
    dataset_path = 'Training.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"[ERROR] Training dataset not found at {dataset_path}")
        
    print(f"[INFO] Loading dataset from '{dataset_path}'...")
    data = pd.read_csv(dataset_path)
    print(f"[INFO] Dataset successfully loaded. Matrix Shape: {data.shape[0]} rows x {data.shape[1]} columns")

    # 2. Feature-Target Matrix Separation
    print("[INFO] Separating independent symptom features (X) and target prognosis (y)...")
    X = data.drop('prognosis', axis=1)
    y = data['prognosis']
    
    feature_count = X.shape[1]
    target_classes = y.nunique()
    print(f"[INFO] Total Symptom Features Extracted: {feature_count}")
    print(f"[INFO] Total Target Disease Classes Identified: {target_classes}")

    # 3. Stratified Dataset Partitioning
    print("[INFO] Splitting dataset into 80% Training and 20% Validation sets (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Training Samples: {X_train.shape[0]} | Validation Samples: {X_test.shape[0]}")

    # 4. Model Instantiation and Fitting
    print("[INFO] Instantiating Random Forest Classifier (n_estimators=100, criterion='gini')...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    print("[INFO] Executing model fitting on training matrix...")
    model.fit(X_train, y_train)
    print("[SUCCESS] Model training complete.")

    # 5. Model Evaluation
    print("[INFO] Evaluating model performance on unseen validation set...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"[RESULT] Overall Classification Accuracy: {accuracy * 100:.2f}%")



    # 6. Artifact Serialization
    print("[INFO] Serializing trained model and symptom feature list for server deployment...")
    model_filename = 'disease_model.pkl'
    symptoms_filename = 'symptoms_list.pkl'
    
    joblib.dump(model, model_filename)
    joblib.dump(list(X.columns), symptoms_filename)
    
    print(f"[SUCCESS] Model binary written to '{model_filename}' ({os.path.getsize(model_filename) / 1024:.2f} KB)")
    print(f"[SUCCESS] Symptom index written to '{symptoms_filename}' ({os.path.getsize(symptoms_filename) / 1024:.2f} KB)")
    print("[INFO] Pipeline execution terminated successfully.")

if __name__ == '__main__':
    main()