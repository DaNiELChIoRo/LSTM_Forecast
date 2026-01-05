#!/bin/bash
# Script to update dependencies in the existing virtual environment
# Use this when requirements.txt changes

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "=========================================="
echo "LSTM Forecast - Update Dependencies"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: Virtual environment not found at: $VENV_DIR"
    echo ""
    echo "Please run the setup script first:"
    echo "  ./setup_venv.sh"
    echo ""
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install/update requirements
echo ""
echo "Updating dependencies from requirements.txt..."
pip install -r "$PROJECT_DIR/requirements.txt" --upgrade

echo ""
echo "=========================================="
echo "Dependencies Updated Successfully!"
echo "=========================================="
echo ""
echo "Installed packages:"
pip list | grep -E "(numpy|pandas|tensorflow|keras|yfinance|scikit-learn)"
echo ""

# Deactivate virtual environment
deactivate
