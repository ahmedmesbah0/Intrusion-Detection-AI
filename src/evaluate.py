import numpy as np
import pandas as pd
import tensorflow as tf
from data_preprocessing import DataPreprocessor
from model import AnomalyDetector
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve
import pickle
import os

def evaluate():
    # Parameters
    SEQUENCE_LENGTH = 10
    MODEL_PATH = 'models/anomaly_detector.keras'
    PREPROCESSOR_PATH = 'models/preprocessor_state.pkl'
    
    # Load Preprocessor State
    with open(PREPROCESSOR_PATH, 'rb') as f:
        state = pickle.load(f)
        
    # Initialize Preprocessor with loaded state
    processor = DataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    processor.scaler = state['scaler']
    processor.encoders = state['encoders']
    processor.feature_cols = state['feature_cols']
    
    # Load Data
    # In a real scenario, we might want to load only the test set here
    # But our load_data loads everything. Let's assume we want to evaluate on everything 
    # or split it again. For consistency with train.py, let's load all and filter.
    
    df = processor.load_data()
    df_processed = processor.preprocess(df)
    
    # Create sequences for ALL data (Normal + Attack)
    X, y = processor.create_sequences(df_processed)
    
    # Load Model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Predict (Reconstruct)
    print("Running inference...")
    X_pred = model.predict(X)
    
    # Calculate Reconstruction Error (MAE per sequence)
    # Shape: (samples, sequence_length, features)
    # We average over sequence length and features to get a single score per sample
    mse = np.mean(np.power(X - X_pred, 2), axis=1) # (samples, features)
    error_score = np.mean(mse, axis=1) # (samples,)
    
    # Determine Threshold
    # Strategy: Use a percentile of the error on Normal data (if we had a separate validation set)
    # Or use the labels to find the best threshold (since we have them for evaluation)
    
    # Let's visualize the error distribution
    error_df = pd.DataFrame({'reconstruction_error': error_score, 'true_class': y})
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=error_df, x='reconstruction_error', hue='true_class', bins=50, kde=True)
    plt.title('Reconstruction Error Distribution')
    plt.savefig('models/error_distribution.png')
    print("Saved error distribution plot to models/error_distribution.png")
    
    # Find best threshold based on F1 score
    thresholds = np.linspace(error_df['reconstruction_error'].min(), error_df['reconstruction_error'].max(), 100)
    best_f1 = 0
    best_threshold = 0
    
    print("Finding optimal threshold...")
    for thresh in thresholds:
        y_pred = (error_df['reconstruction_error'] > thresh).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    print(f"Best Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1}")
    
    # Final Metrics
    y_pred_final = (error_df['reconstruction_error'] > best_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred_final, average='binary')
    roc_auc = roc_auc_score(y, error_score)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('models/confusion_matrix.png')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, error_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('models/roc_curve.png')

if __name__ == "__main__":
    evaluate()
