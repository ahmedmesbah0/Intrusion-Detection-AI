import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os

class DataPreprocessor:
    def __init__(self, data_dir='dataset_kaggle', sequence_length=10):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.encoders = {}
        self.feature_cols = None

    def load_data(self, train_file='UNSW_NB15_training-set.csv', test_file='UNSW_NB15_testing-set.csv'):
        train_path = os.path.join(self.data_dir, train_file)
        test_path = os.path.join(self.data_dir, test_file)

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Data files not found in {self.data_dir}. Please ensure {train_file} and {test_file} exist.")

        print("Loading data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Combine for consistent preprocessing
        full_df = pd.concat([train_df, test_df], axis=0)
        return full_df

    def preprocess(self, df):
        print("Preprocessing data...")
        # Drop ID and potentially label columns for feature extraction
        # We keep 'label' or 'attack_cat' for evaluation later, but separate them now
        
        drop_cols = ['id']
        if 'id' in df.columns:
            df = df.drop(columns=drop_cols)
            
        # Identify categorical and numerical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        num_cols = df.select_dtypes(include=['number']).columns
        
        # Exclude target variables from scaling/encoding if present
        targets = ['label', 'attack_cat']
        feature_cols = [c for c in df.columns if c not in targets]
        
        self.feature_cols = feature_cols
        
        # Encode categorical features
        for col in cat_cols:
            if col in feature_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                
        # Scale numerical features
        # We fit on the whole dataset for simplicity in this demo, 
        # but in strict production, fit on train, transform on test.
        # Here we assume the input 'df' is the training set or we handle split later.
        
        # Filter only feature columns for scaling
        df_features = df[feature_cols]
        df_features = self.scaler.fit_transform(df_features)
        
        # Reconstruct DataFrame
        df_processed = pd.DataFrame(df_features, columns=feature_cols)
        
        # Add back targets if they existed
        for target in targets:
            if target in df.columns:
                df_processed[target] = df[target].values
                
        return df_processed

    def create_sequences(self, df, label_col='label'):
        """
        Creates sequences for LSTM.
        df: Preprocessed DataFrame
        """
        X = []
        y = []
        
        # We only use features for X
        feature_data = df[self.feature_cols].values
        labels = df[label_col].values if label_col in df.columns else None
        
        print(f"Creating sequences of length {self.sequence_length}...")
        for i in range(len(df) - self.sequence_length):
            X.append(feature_data[i : i + self.sequence_length])
            if labels is not None:
                # Label for the sequence is the label of the last step (or majority, etc.)
                # Here we take the label of the last step
                y.append(labels[i + self.sequence_length - 1])
                
        return np.array(X), np.array(y)

if __name__ == "__main__":
    # Example usage
    try:
        processor = DataPreprocessor()
        df = processor.load_data()
        df_clean = processor.preprocess(df)
        X, y = processor.create_sequences(df_clean)
        print(f"Data shape: {X.shape}")
    except FileNotFoundError as e:
        print(e)
