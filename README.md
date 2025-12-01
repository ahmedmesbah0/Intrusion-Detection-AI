# Intrusion Detection System: CNN+LSTM Autoencoder

A student-level cybersecurity project for detecting network intrusions using the UNSW-NB15 dataset with a hybrid 1D Convolutional Neural Network (CNN) + LSTM Autoencoder architecture.

## 🎯 Project Goal

Build an intelligent intrusion detection system that can automatically recognize abnormal or malicious network activity by analyzing sequential patterns of network flows. The system uses deep learning to model normal network behavior and detect anomalies through reconstruction error analysis.

## 📊 Dataset

**UNSW-NB15 Dataset** (Available on Kaggle)
- Contains 49 features per network flow
- Includes packet count, byte rate, flags, protocol information, etc.
- Traffic includes normal and nine attack categories:
  - Fuzzers
  - DoS (Denial of Service)
  - Exploits
  - Reconnaissance
  - Shellcode
  - Analysis
  - Backdoor
  - Generic
  - Worms

## 🏗️ Architecture

The project uses a hybrid deep learning approach:

1. **1D CNN Layers**: Extract spatial correlations among packet/flow features
2. **LSTM Autoencoder**: Capture temporal dependencies and reconstruct normal network behavior
3. **Anomaly Detection**: Identify sequences with high reconstruction error as potential intrusions

## 📁 Project Structure

```
Intrusion-Detection-AI/
├── data/                          # Dataset directory (UNSW-NB15 CSV files)
├── saved_models/                  # Trained model checkpoints
├── preprocessed_data/             # Intermediate processed files
├── 1_preprocessing.ipynb          # Data loading and preprocessing
├── 2_visualization.ipynb          # Exploratory data analysis
├── 3_model_training.ipynb         # Model architecture and training
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Kaggle account (for dataset download)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Intrusion-Detection-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the UNSW-NB15 Dataset**:
   - Visit [UNSW-NB15 on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
   - Download the dataset files
   - Place the CSV files in the `data/` directory:
     - `UNSW_NB15_training-set.csv`
     - `UNSW_NB15_testing-set.csv`

### Usage

Run the notebooks in order:

1. **Preprocessing** (`1_preprocessing.ipynb`):
   ```bash
   jupyter notebook 1_preprocessing.ipynb
   ```
   - Loads and cleans the dataset
   - Engineers features and creates sequences
   - Saves preprocessed data for subsequent notebooks

2. **Visualization** (`2_visualization.ipynb`):
   ```bash
   jupyter notebook 2_visualization.ipynb
   ```
   - Performs exploratory data analysis
   - Visualizes feature distributions and attack patterns
   - Creates correlation heatmaps and insights

3. **Model Training** (`3_model_training.ipynb`):
   ```bash
   jupyter notebook 3_model_training.ipynb
   ```
   - Builds the CNN+LSTM Autoencoder model
   - Trains on normal traffic patterns
   - Evaluates intrusion detection performance
   - Generates comprehensive results and visualizations

## 📈 Expected Results

The model is expected to achieve:
- **Accuracy**: >80% on test set
- **Precision/Recall**: Balanced detection of various attack types
- **AUC-ROC**: >0.85 for binary classification (normal vs attack)

## 📚 Learning Objectives

Through this project, you will learn:
- How to preprocess flow-based network features
- Creating meaningful sequences from network traffic data
- Applying 1D CNNs for spatial feature extraction
- Using LSTM Autoencoders for temporal dependency modeling
- Anomaly detection through reconstruction error analysis
- Evaluating cybersecurity models with appropriate metrics

## 🛠️ Technical Details

- **Framework**: TensorFlow/Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Seaborn, Matplotlib
- **Model Type**: Unsupervised anomaly detection (trained on normal traffic)

## 📝 Documentation

Each Jupyter notebook contains detailed markdown cells explaining:
- The purpose of each step
- Technical concepts and terminology
- Code explanations and comments
- Interpretation of results

## 🤝 Contributing

This is a student-level educational project. Feel free to:
- Experiment with different architectures
- Try alternative preprocessing techniques
- Adjust hyperparameters
- Extend the visualization analysis

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- UNSW-NB15 Dataset creators
- TensorFlow and Keras communities
- Cybersecurity research community
