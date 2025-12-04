#!/bin/bash

# RAG Workshop Setup Script
# This script sets up your environment for the workshop

set -e  # Exit on error

echo "🚀 RAG Workshop - Environment Setup"
echo "===================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found: Python $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   ⚠️  Virtual environment already exists"
    read -p "   Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo "   ✅ Virtual environment recreated"
    else
        echo "   ℹ️  Using existing virtual environment"
    fi
else
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q
echo "   ✅ pip upgraded"

# Install dependencies
echo ""
echo "📚 Installing dependencies from requirements.txt..."
pip install -r requirements.txt -q
echo "   ✅ Dependencies installed"

# Verify installations
echo ""
echo "🔍 Verifying installations..."

verify_package() {
    if python -c "import $1" 2>/dev/null; then
        echo "   ✅ $2"
    else
        echo "   ❌ $2 - FAILED"
        return 1
    fi
}

verify_package "google.cloud.secretmanager" "Secret Manager"
verify_package "psycopg2" "PostgreSQL"
verify_package "pgvector.psycopg2" "pgvector"
python -c "from google import genai" 2>/dev/null && echo "   ✅ Vertex AI (google-genai)" || echo "   ❌ Vertex AI - FAILED"
verify_package "pymupdf4llm" "PyMuPDF4LLM"
verify_package "langchain_text_splitters" "LangChain"

# Check Jupyter
echo ""
if command -v jupyter &> /dev/null; then
    echo "   ✅ Jupyter already installed"
else
    echo "📓 Installing Jupyter..."
    pip install jupyter jupyterlab -q
    echo "   ✅ Jupyter installed"
fi

# GCP Authentication check
echo ""
echo "🔐 Checking GCP authentication..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "   ⚠️  gcloud CLI not found"
    echo "   ℹ️  Install from: https://cloud.google.com/sdk/docs/install"
    echo "   ℹ️  After installing, run:"
    echo "      gcloud auth application-default login"
    echo "      gcloud config set project data-science-faggruppe-rag"
elif gcloud auth application-default print-access-token &>/dev/null; then
    echo "   ✅ GCP authentication configured"
else
    echo "   ⚠️  GCP not authenticated"
    echo ""
    read -p "   Authenticate now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        gcloud auth application-default login
        gcloud config set project data-science-faggruppe-rag
        echo "   ✅ GCP authentication complete"
    else
        echo "   ℹ️  Remember to run: gcloud auth application-default login"
    fi
fi

# Final instructions
echo ""
echo "✨ Setup Complete!"
echo "=================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment (if not already active):"
echo "   source venv/bin/activate"
echo ""
echo "2. Get Cloud SQL IP address:"
echo "   gcloud sql instances describe vector-db-instance \\"
echo "     --format='value(ipAddresses[0].ipAddress)'"
echo ""
echo "3. Start Jupyter:"
echo "   jupyter notebook src/notebook/intro.ipynb"
echo ""
echo "4. In the notebook:"
echo "   - Make sure to select the 'venv' kernel"
echo "   - Update DB_HOST with the Cloud SQL IP"
echo ""
echo "Happy learning! 🎉"
