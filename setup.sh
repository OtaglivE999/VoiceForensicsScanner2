#!/usr/bin/env bash
#
# Voice Forensics Scanner - Linux Setup & Launch
# Usage: bash setup.sh [--install-only]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "==============================================================="
echo "  VOICE FORENSICS SCANNER - Setup"
echo "==============================================================="
echo ""

# Check Python version
PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"
            echo "[OK]   Found $cmd ($version)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "[FAIL] Python 3.10+ is required but not found."
    echo "       Install Python 3.10+ and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    "$PYTHON_CMD" -m venv venv
    echo "[OK]   Virtual environment created."
fi

# Activate virtual environment
source venv/bin/activate
echo "[OK]   Virtual environment activated."

# Install/upgrade pip
echo ""
echo "Ensuring pip is up to date..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing required modules..."
pip install -r requirements.txt
echo ""
echo "[OK]   All requirements installed."

# Verify modules
echo ""
"$PYTHON_CMD" verify_install.py

if [ "$1" = "--install-only" ]; then
    echo ""
    echo "Setup complete. Run 'bash setup.sh' again without --install-only to launch."
    exit 0
fi

# Copy .env if needed
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo ""
    echo "[OK]   Created .env from .env.example (edit as needed)."
fi

# List audio devices and launch
echo ""
echo "Listing available audio devices..."
echo ""
"$PYTHON_CMD" src/soft_voice_scanner.py --list-devices

echo ""
echo "==============================================================="
echo "  Enter device number to start scanning (or 'q' to quit):"
echo "==============================================================="
read -rp "Device ID: " DEVICE_ID

if [ "$DEVICE_ID" = "q" ] || [ "$DEVICE_ID" = "Q" ]; then
    echo ""
    echo "Scanner stopped."
    exit 0
fi

echo ""
echo "Starting scanner on device $DEVICE_ID..."
echo "Press Ctrl+C to stop"
echo ""

"$PYTHON_CMD" src/soft_voice_scanner.py --device "$DEVICE_ID" --ultra-sensitive
