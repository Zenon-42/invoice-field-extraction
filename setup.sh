#!/bin/bash

echo "=================================================="
echo "  Invoice Extraction System - Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "🔍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "❌ Error: Python 3.8 or higher required"
    exit 1
fi

echo "✅ Python version OK"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "=================================================="
echo "  Next Steps:"
echo "=================================================="
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set your Anthropic API key (optional, for VLM mode):"
echo "   export ANTHROPIC_API_KEY='your_api_key_here'"
echo ""
echo "3. Run the extraction pipeline:"
echo "   python executable.py --input invoice.pdf --output result.json --use-api"
echo ""
echo "4. For batch processing:"
echo "   python executable.py --input ./invoices/ --output results.json --use-api"
echo ""
echo "5. For CPU-only mode (no API costs):"
echo "   python executable.py --input invoice.pdf --output result.json"
echo ""
echo "=================================================="
echo "  Testing Installation"
echo "=================================================="
echo ""
echo "Running quick test..."

# Test imports
python3 -c "
import sys
try:
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    from PIL import Image
    print('✅ Core dependencies loaded successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "   You're ready to process invoices!"
else
    echo ""
    echo "⚠️  Some dependencies failed to load"
    echo "   Please check the error messages above"
fi

echo ""
echo "=================================================="
