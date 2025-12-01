# Quick Start Guide - Running the Jupyter Notebooks

## 🚀 Quick Setup (3 Steps)

### Step 1: Install System Dependencies

First, install the required system packages:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
```

### Step 2: Run the Setup Script

We've provided an automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

**OR** do it manually:

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Launch Jupyter Notebook

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

This will open Jupyter in your web browser!

---

## 📓 Running the Notebooks

Once Jupyter is open, run the notebooks in this order:

1. **`1_preprocessing.ipynb`** - First time only (preprocesses the dataset)
2. **`2_visualization.ipynb`** - Explore the data
3. **`3_model_training.ipynb`** - Train and evaluate the model

---

## 📥 Download the Dataset

Before running the notebooks, download the UNSW-NB15 dataset:

1. Go to: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
2. Download the dataset files
3. Place these files in the `data/` directory:
   - `UNSW_NB15_training-set.csv`
   - `UNSW_NB15_testing-set.csv`

---

## 🔄 Daily Usage

After initial setup, each time you want to work on the project:

```bash
# Navigate to project directory
cd /home/mesbah7/Github/Repos/Intrusion-Detection-AI

# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook
```

When you're done:

```bash
# Stop Jupyter (Ctrl+C in terminal)
# Deactivate virtual environment
deactivate
```

---

## 🐛 Troubleshooting

### Issue: "externally-managed-environment" error
**Solution**: Always use the virtual environment! Run `source venv/bin/activate` first.

### Issue: "ModuleNotFoundError"
**Solution**: Make sure you've installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Jupyter not opening
**Solution**: Try specifying the browser:
```bash
jupyter notebook --browser=firefox
# or
jupyter notebook --browser=chrome
```

### Issue: Can't find dataset
**Solution**: Make sure you've placed the CSV files in the `data/` directory

---

## 🎯 Expected File Structure

After setup and downloading the dataset:

```
Intrusion-Detection-AI/
├── venv/                          ← Virtual environment
├── data/                          ← Your downloaded dataset
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── 1_preprocessing.ipynb
├── 2_visualization.ipynb
├── 3_model_training.ipynb
├── requirements.txt
└── setup.sh
```

---

## ✅ Verification

To verify everything is set up correctly:

```bash
source venv/bin/activate
python -c "import tensorflow, pandas, seaborn; print('✅ All libraries installed!')"
```

You should see: `✅ All libraries installed!`
