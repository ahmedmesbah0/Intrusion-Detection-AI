# Quick Start Guide

## Quick Setup (3 Steps)

### Step 1: Install System Dependencies

First, install the required packages:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

### Step 2: Run the Setup Script

We have an automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

Or if you want to do it manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Launch Jupyter

```bash
# Make sure virtual environment is active
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

This will open Jupyter in your browser.

---

## Running the Notebooks

Once Jupyter opens, run the notebooks in order:

1. `1_preprocessing.ipynb` - Run this first (preprocesses the dataset)
2. `2_visualization.ipynb` - Look at the data
3. `3_model_training.ipynb` - Train and test the model

---

## Download the Dataset

Before running the notebooks, get the UNSW-NB15 dataset:

1. Go to: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
2. Download the files
3. Put them in the `dataset_kaggle/` folder:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

---

## Daily Use

After the initial setup, each time you work on the project:

```bash
# Go to project folder
cd /home/mesbah7/Github/Repos/Intrusion-Detection-AI

# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

When you're done:

```bash
# Stop Jupyter (press Ctrl+C in terminal)
# Deactivate virtual environment
deactivate
```

---

## Troubleshooting

### Problem: "externally-managed-environment" error
**Fix**: Use the virtual environment. Run `source venv/bin/activate` first.

### Problem: "ModuleNotFoundError"
**Fix**: Install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: Jupyter won't open
**Fix**: Try specifying the browser:
```bash
jupyter notebook --browser=firefox
# or
jupyter notebook --browser=chrome
```

### Problem: Can't find dataset
**Fix**: Make sure you put the CSV files in the `dataset_kaggle/` folder

---

## File Structure

After setup and downloading the dataset, you should have:

```
Intrusion-Detection-AI/
├── venv/                          # Virtual environment
├── dataset_kaggle/                # Downloaded dataset
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── 1_preprocessing.ipynb
├── 2_visualization.ipynb
├── 3_model_training.ipynb
├── requirements.txt
└── setup.sh
```

---

## Check Everything Works

To check if everything is set up right:

```bash
source venv/bin/activate
python -c "import tensorflow, pandas, seaborn; print('All libraries installed')"
```

You should see: `All libraries installed`
