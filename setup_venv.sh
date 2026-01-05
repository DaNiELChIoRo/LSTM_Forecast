#!/bin/bash
# Virtual Environment Setup Script for LSTM Forecast Project
# This script creates an isolated Python environment to avoid conflicts with other projects

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "=========================================="
echo "LSTM Forecast - Virtual Environment Setup"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"
echo ""

# Remove existing virtual environment if it exists
if [ -d "$VENV_DIR" ]; then
    echo "Existing virtual environment found. Removing..."
    rm -rf "$VENV_DIR"
fi

# Create new virtual environment
echo "Creating virtual environment in: $VENV_DIR"
python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip to latest version
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing project dependencies from requirements.txt..."
echo "This may take several minutes..."
pip install -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment created at: $VENV_DIR"
echo ""
echo "To activate the virtual environment manually, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To deactivate when done, run:"
echo "  deactivate"
echo ""
echo "To run your project with the virtual environment:"
echo "  ./run_with_venv.sh"
echo ""
