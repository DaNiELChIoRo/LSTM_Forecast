#!/bin/bash
# Script to run main.py with the project's virtual environment
# Safe to use in Jenkins or manual execution

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo "=========================================="
echo "LSTM Forecast - Running with Virtual Env"
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
echo "Activating virtual environment: $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Verify we're using the correct Python
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Run the main script
echo "Running main.py..."
echo "=========================================="
echo ""
python main.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Script completed successfully!"
else
    echo "Script failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

# Deactivate virtual environment
deactivate

exit $EXIT_CODE
