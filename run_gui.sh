#!/bin/bash
# Launch the Py-Feat GUI
# Usage: ./run_gui.sh [--port PORT]

set -e

PORT="${1:-8501}"
if [ "$1" = "--port" ]; then
    PORT="$2"
fi

echo "Starting Py-Feat GUI on http://localhost:${PORT}"
echo "Press Ctrl+C to stop."
echo ""

# Activate the pyfeat conda env if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate pyfeat 2>/dev/null || true
fi

streamlit run gui_app.py --server.port "$PORT" --server.headless true
