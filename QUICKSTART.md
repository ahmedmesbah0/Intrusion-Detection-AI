# Quick Start Guide

## Setup

### 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

### 3. Launch Jupyter

```bash
source venv/bin/activate
jupyter notebook
```

## Run Notebooks

Once Jupyter opens, run the notebooks in order:

1.  `1_preprocessing.ipynb`
2.  `2_visualization.ipynb`
3.  `3_model_training.ipynb`

## Download Dataset

1.  Go to: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
2.  Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
3.  Place them in the `dataset_kaggle/` folder.
