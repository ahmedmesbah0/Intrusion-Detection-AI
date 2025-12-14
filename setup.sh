#!/bin/bash

# Setup Script for Intrusion Detection Project
# This script sets up the Python environment and installs all dependencies

echo "================================================"
echo "Intrusion Detection System - Setup Script"
echo "================================================"

# Step 1: Install system dependencies
echo ""
echo "Step 1: Installing system dependencies..."
echo "You may be prompted for your password."
sudo apt update
sudo apt install -y python3-venv python3-pip

# Step 2: Create virtual environment
echo ""
echo "Step 2: Creating virtual environment..."
python3 -m venv venv

# Step 3: Activate virtual environment and upgrade pip
echo ""
echo "Step 3: Activating virtual environment and upgrading pip..."
source venv/bin/activate
pip install --upgrade pip

# Step 4: Install project dependencies
echo ""
echo "Step 4: Installing project dependencies..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run Jupyter notebooks, run:"
echo "  jupyter notebook"
echo ""
echo "To deactivate the virtual environment, run:"
echo "  deactivate"
echo ""
