#!/bin/bash
# =============================================================================
# AmpereData Streamlit - Standalone Startup Script
# =============================================================================
# Starts only the Streamlit application
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMLIT_PORT=8501

echo "Starting AmpereData Streamlit on port $STREAMLIT_PORT..."

cd "$SCRIPT_DIR"

streamlit run streamlit_app.py --server.port=$STREAMLIT_PORT
