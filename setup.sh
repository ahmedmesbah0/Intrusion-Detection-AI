#!/bin/bash

echo "========================================"
echo "Intrusion Detection System - Setup"
echo "========================================"

# Install system dependencies
echo ""
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-venv python3-pip

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and upgrade pip
echo ""
echo "Activating environment..."
source venv/bin/activate
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate: source venv/bin/activate"
echo "To run: jupyter notebook"
echo "To deactivate: deactivate"
echo ""
