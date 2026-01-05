# LSTM Forecast - Virtual Environment Setup Guide

This guide explains how to set up and use an isolated Python virtual environment for the LSTM Forecast project. Using a virtual environment ensures that this project's dependencies don't conflict with other Python projects on your server.

## Quick Start

### First Time Setup

1. Run the setup script to create the virtual environment:
   ```bash
   ./setup_venv.sh
   ```

2. The script will:
   - Create a new virtual environment in the `venv/` directory
   - Install all required dependencies from `requirements.txt`
   - Use NumPy < 2.0 to ensure compatibility

### Running the Project

```bash
./run_with_venv.sh
```

This automatically activates the virtual environment, runs `main.py`, and deactivates when done.

## Available Scripts

### `setup_venv.sh`
Creates a fresh virtual environment and installs all dependencies.

**When to use:**
- First time setting up the project
- After corrupting the virtual environment
- When you want a completely clean installation

**Usage:**
```bash
./setup_venv.sh
```

### `run_with_venv.sh`
Runs the main.py script using the project's virtual environment.

**When to use:**
- Every time you want to run the project
- In Jenkins jobs (replace direct python calls with this script)

**Usage:**
```bash
./run_with_venv.sh
```

### `update_dependencies.sh`
Updates dependencies in the existing virtual environment.

**When to use:**
- After modifying `requirements.txt`
- When you want to upgrade packages without recreating the entire environment

**Usage:**
```bash
./update_dependencies.sh
```

## Jenkins Integration

### Update Your Jenkins Job Configuration

Replace your current shell command with:

```bash
#!/bin/bash
set -e

# Navigate to project directory
cd /Users/danielmenesesleon/PycharmProjects/LSTM_Forecast

# Run with virtual environment
./run_with_venv.sh
```

### First Time Jenkins Setup

Before running the Jenkins job for the first time, SSH into your Jenkins server and run:

```bash
cd /Users/danielmenesesleon/PycharmProjects/LSTM_Forecast
./setup_venv.sh
```

This creates the virtual environment that Jenkins will use.

## Manual Virtual Environment Usage

If you prefer to work manually with the virtual environment:

### Activate
```bash
source venv/bin/activate
```

### Verify it's active
```bash
which python
# Should show: /Users/danielmenesesleon/PycharmProjects/LSTM_Forecast/venv/bin/python

python --version
```

### Run your code
```bash
python main.py
```

### Deactivate when done
```bash
deactivate
```

## Troubleshooting

### Virtual environment not found
**Error:** `Virtual environment not found at: .../venv`

**Solution:** Run `./setup_venv.sh` first to create the environment.

### NumPy version issues
**Error:** `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**Solution:**
1. Delete the virtual environment: `rm -rf venv/`
2. Re-run setup: `./setup_venv.sh`
3. The updated `requirements.txt` now pins NumPy < 2.0

### Permission denied
**Error:** `Permission denied: ./setup_venv.sh`

**Solution:** Make scripts executable:
```bash
chmod +x setup_venv.sh run_with_venv.sh update_dependencies.sh
```

### Packages not installing
**Error:** Various pip installation errors

**Solution:**
1. Ensure you have internet connectivity
2. Try upgrading pip manually:
   ```bash
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Dependency Information

The project now uses NumPy 1.x (< 2.0) to ensure compatibility with:
- TensorFlow >= 2.10.0
- Keras >= 2.10.0
- pandas >= 1.4.0
- h5py >= 3.7.0
- numexpr >= 2.8.0
- bottleneck >= 1.3.0

These versions are pinned in `requirements.txt` to prevent compatibility issues.

## Best Practices

1. **Always use the virtual environment** - Never install packages globally
2. **Update dependencies properly** - Use `update_dependencies.sh` after changing `requirements.txt`
3. **Don't commit venv/** - The `venv/` directory is project-specific and should be in `.gitignore`
4. **Document changes** - If you add new dependencies, update `requirements.txt` and run `update_dependencies.sh`

## Additional Notes

- The virtual environment is stored in the `venv/` directory within your project
- Each project can have its own isolated environment with different dependency versions
- The virtual environment folder is typically 500MB-1GB in size
- Virtual environments are recreatable - you can always delete and recreate them using `setup_venv.sh`
