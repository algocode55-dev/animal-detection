#!/bin/bash
# Startup script for Animal Detection Dashboard

# Resolve script directory to allow running from anywhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set up local library overrides for missing X11/Qt libraries
export LD_LIBRARY_PATH="$SCRIPT_DIR/lib_override/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Set up Qt platform plugin path to avoid OpenCV conflict
export QT_QPA_PLATFORM_PLUGIN_PATH="$SCRIPT_DIR/.venv/lib/python3.12/site-packages/PyQt6/Qt6/plugins"

# Run the PyQt6 application
echo "Starting AI Animal Roadway Detection System..."
.venv/bin/python dashboard.py "$@"
