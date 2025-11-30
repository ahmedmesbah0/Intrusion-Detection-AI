import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, TimeDistributed, RepeatVector, Dropout

class AnomalyDetector:
    def __init__(self, sequence_length, num_features):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = self._build_model()

    def _build_model(self):
        # Input Layer
        inputs = Input(shape=(self.sequence_length, self.num_features))

        # Encoder
        # 1D CNN for spatial feature extraction
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(inputs)
        x = Dropout(0.2)(x)
        
        # LSTM for temporal dependencies
        # Return sequences=False to get a fixed vector representation
        x = LSTM(64, activation='relu', return_sequences=False)(x)
        
        # Latent representation (bottleneck)
        x = RepeatVector(self.sequence_length)(x)

        # Decoder
        # LSTM to reconstruct the sequence
        x = LSTM(64, activation='relu', return_sequences=True)(x)
        x = Dropout(0.2)(x)
        
        # Output Layer
        # TimeDistributed Dense to map back to feature space for each time step
        output = TimeDistributed(Dense(self.num_features))(x)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        
        return model

    def summary(self):
        self.model.summary()

    def train(self, X_train, epochs=10, batch_size=32, validation_split=0.1):
        history = self.model.fit(
            X_train, X_train, # Autoencoder target is input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        return tf.keras.models.load_model(path)

if __name__ == "__main__":
    # Test model build
    model = AnomalyDetector(sequence_length=10, num_features=40)
    model.summary()
