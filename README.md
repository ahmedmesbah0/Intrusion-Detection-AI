# Intrusion Detection System: CNN+LSTM Autoencoder

Network intrusion detection using UNSW-NB15 dataset with hybrid 1D CNN + LSTM Autoencoder.

## Project Discussion Requirements

**Schedule**: Monday, December 15th, 2025 | 9:00 AM - 2:00 PM

**Locations**:
- Instructor (Dr. Ahmed Bayumi): Office B7.F2.15
- TAs: Meeting Room B7.F2.20

**Format**: 3-minute presentation per team. Ensure laptop and model are ready.

**Grading** (10 points total):
1. Methodology/Design - 2 pts
2. Implementation Quality - 2 pts
3. Results/Evaluation - 2 pts
4. Innovation/Complexity - 2 pts
5. Report Quality - 1 pt
6. Presentation/Defense - 1 pt

## Overview

Detects network intrusions using deep learning to identify anomalous traffic patterns based on reconstruction error.

**Dataset**: UNSW-NB15 (Kaggle)
- 49 features per flow (packet count, byte rate, flags, protocol info)
- Normal traffic + 9 attack types (Fuzzers, DoS, Exploits, Reconnaissance, Shellcode, Analysis, Backdoor, Generic, Worms)

**Architecture**:
1. 1D CNN - Spatial feature extraction
2. LSTM Autoencoder - Temporal pattern learning
3. Anomaly detection via reconstruction error

## Project Structure

```
Intrusion-Detection-AI/
├── dataset_kaggle/                # Dataset directory (UNSW-NB15 CSV files)
├── saved_models/                  # Trained model checkpoints
├── preprocessed_data/             # Processed data files
├── 1_preprocessing.ipynb          # Data loading and preprocessing
├── 2_visualization.ipynb          # Data visualization and analysis
├── 3_model_training.ipynb         # Model training and evaluation
├── requirements.txt               # Python dependencies
├── setup.sh                       # Setup script
└── README.md                      # This file
```

## Setup

**Requirements**: Python 3.8+, Jupyter, Kaggle account

**Installation**:
```bash
git clone <repository-url>
cd Intrusion-Detection-AI
chmod +x setup.sh
./setup.sh
```

**Dataset**: Download from [Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15) and place in `dataset_kaggle/`

## Usage

Run notebooks in order: `1_preprocessing.ipynb` → `2_visualization.ipynb` → `3_model_training.ipynb`

**Expected Performance**:
- Accuracy: >80%
- AUC-ROC: >0.85

## Technical Stack

TensorFlow/Keras, Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib

## License

Educational use only.
