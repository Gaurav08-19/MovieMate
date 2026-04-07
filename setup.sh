#!/bin/bash
set -e

echo "=== MovieMate Setup ==="
echo ""

# Create directory structure
mkdir -p data src

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Check for API_KEY
if [ -z "$API_KEY" ]; then
    echo ""
    echo "WARNING: API_KEY is not set."
    echo "  Set it before running:"
    echo "    export API_KEY='your-key-here'"
else
    echo "API_KEY detected."
fi

# Register kernel for Jupyter
python -m ipykernel install --user --name moviemate --display-name "MovieMate (Python 3)"

touch src/__init__.py

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Activate the environment:  source venv/bin/activate"
echo "Launch Gradio app:         python app.py"
echo "Launch Jupyter notebook:   jupyter notebook MovieMate_Notebook.ipynb"
