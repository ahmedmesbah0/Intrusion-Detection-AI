import numpy as np
import pandas as pd
import tensorflow as tf
from data_preprocessing import DataPreprocessor
from model import AnomalyDetector
import os
import pickle

def train():
    # Parameters
    SEQUENCE_LENGTH = 10
    EPOCHS = 10
    BATCH_SIZE = 32
    MODEL_SAVE_PATH = 'models/anomaly_detector.keras'
    
    # Ensure models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load and Preprocess Data
    processor = DataPreprocessor(sequence_length=SEQUENCE_LENGTH)
    try:
        df = processor.load_data()
        df_processed = processor.preprocess(df)
        
        # Save the scaler and encoders for later use during inference
        with open('models/preprocessor_state.pkl', 'wb') as f:
            pickle.dump({'scaler': processor.scaler, 'encoders': processor.encoders, 'feature_cols': processor.feature_cols}, f)
            
        # Filter for Normal traffic only for training
        # Assuming 'label' 0 is normal, 1 is attack
        # Or 'attack_cat' == 'Normal'
        if 'attack_cat' in df_processed.columns:
             # We need to inverse transform to check 'Normal' if encoded, 
             # but simpler: use the label column if 0=Normal
             pass
        
        # For simplicity, let's assume label 0 is normal.
        # In the dummy data, 0 is normal.
        
        df_normal = df_processed[df_processed['label'] == 0]
        print(f"Training on {len(df_normal)} normal samples.")
        
        # Create sequences
        X_train, _ = processor.create_sequences(df_normal)
        
        # Build Model
        num_features = X_train.shape[2]
        detector = AnomalyDetector(sequence_length=SEQUENCE_LENGTH, num_features=num_features)
        detector.summary()
        
        # Train
        print("Starting training...")
        history = detector.train(X_train, epochs=EPOCHS, batch_size=BATCH_SIZE)
        
        # Save Model
        detector.save(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train()
