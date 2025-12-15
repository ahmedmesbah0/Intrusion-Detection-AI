# Intrusion Detection System: CNN+LSTM Autoencoder

A cybersecurity project for detecting network intrusions using the UNSW-NB15 dataset with a hybrid 1D CNN + LSTM Autoencoder architecture.

## Project Requirements

### Discussion Schedule
- **Date**: Monday, December 15th, 2025
- **Time**: 9:00 AM to 2:00 PM
- **Locations**:
  - Instructor (Dr. Ahmed Bayumi): Office B7.F2.15
  - TAs: Meeting Room B7.F2.20

### Discussion Format
- Each team has 3 minutes to present their idea and approach
- Make sure your laptop is working and the model/program is running before the discussion
- Be ready to show the working system

### Grading Distribution (Total: 10 points)

1. **Methodology / System Design** - 2 points
2. **Implementation & Technical Quality** - 2 points
3. **Results & Evaluation** - 2 points
4. **Innovation / Complexity** - 2 points
5. **Report Quality** - 1 point
6. **Presentation & Defense** - 1 point

## Project Goal

This project builds an intrusion detection system that can find abnormal or malicious network activity by looking at patterns in network traffic. We use deep learning to learn what normal network behavior looks like, then detect anomalies by checking how well the model can reconstruct the data.

## Dataset

**UNSW-NB15 Dataset** (available on Kaggle)
- Has 49 features per network flow
- Includes things like packet count, byte rate, flags, protocol info, etc.
- Traffic includes normal traffic and nine attack types:
  - Fuzzers
  - DoS (Denial of Service)
  - Exploits
  - Reconnaissance
  - Shellcode
  - Analysis
  - Backdoor
  - Generic
  - Worms

## Architecture

The project uses a hybrid deep learning approach:

1. **1D CNN Layers**: Extracts spatial correlations from packet/flow features
2. **LSTM Autoencoder**: Captures temporal patterns and reconstructs normal network behavior
3. **Anomaly Detection**: Flags sequences with high reconstruction error as potential intrusions

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

## Getting Started

### What You Need

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Kaggle account (to download the dataset)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Intrusion-Detection-AI
   ```

2. Run the setup script (recommended):
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

   Or install manually:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Download the UNSW-NB15 Dataset:
   - Go to [UNSW-NB15 on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
   - Download the dataset
   - Place the CSV files in the `dataset_kaggle/` folder:
     - `UNSW_NB15_training-set.csv`
     - `UNSW_NB15_testing-set.csv`

### How to Use

Run the notebooks in this order:

1. **Preprocessing** (`1_preprocessing.ipynb`):
   ```bash
   jupyter notebook 1_preprocessing.ipynb
   ```
   - Loads and cleans the dataset
   - Does feature engineering and creates sequences
   - Saves preprocessed data for the next notebooks

2. **Visualization** (`2_visualization.ipynb`):
   ```bash
   jupyter notebook 2_visualization.ipynb
   ```
   - Does exploratory data analysis
   - Shows feature distributions and attack patterns
   - Creates correlation heatmaps

3. **Model Training** (`3_model_training.ipynb`):
   ```bash
   jupyter notebook 3_model_training.ipynb
   ```
   - Builds the CNN+LSTM Autoencoder model
   - Trains on normal traffic patterns
   - Tests the intrusion detection performance
   - Shows results and visualizations

## Expected Results

The model should get:
- **Accuracy**: Over 80% on test set
- **Precision/Recall**: Good detection across different attack types
- **AUC-ROC**: Over 0.85 for binary classification (normal vs attack)

## What This Project Covers

- How to preprocess network flow features
- Creating sequences from network traffic data
- Using 1D CNNs for feature extraction
- Using LSTM Autoencoders for temporal modeling
- Anomaly detection with reconstruction errors
- Evaluating cybersecurity models

## Technical Stack

- **Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Seaborn, Matplotlib
- **Model Type**: Unsupervised anomaly detection (trained on normal traffic only)

## Documentation

Each notebook has markdown cells that explain:
- What each step does
- Technical concepts and terms
- How the code works
- What the results mean

## License

This project is for educational purposes.

## Credits

- UNSW-NB15 Dataset creators
- TensorFlow and Keras communities
- Cybersecurity research community
