#!/bin/bash
# ============================================================
#  MTA for Master — First-time install (Mac/Linux)
# ============================================================
#  This script downloads an isolated Python and installs
#  everything MTA needs, inside the folder it sits in. It does
#  not touch anything else on your computer.
#  Duration: 5 to 10 minutes depending on your connection.
# ============================================================

# Move into the script's directory (works on Mac and Linux)
cd "$(dirname "$0")" || exit 1

echo ""
echo "============================================================"
echo "  MTA — Installation"
echo "============================================================"
echo ""
echo "  What will happen:"
echo "    1. Download uv (Python manager, ~30 MB)"
echo "    2. Download isolated Python 3.12 (~50 MB)"
echo "    3. Install scientific libraries (~400 MB)"
echo ""
echo "  EVERYTHING stays inside this folder. To uninstall later,"
echo "  simply delete this entire folder."
echo ""
echo "  Estimated duration: 5 to 10 minutes."
echo ""
read -r -p "  Press Enter to continue..." _

# Detect OS and architecture for uv
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS-$ARCH" in
    "Darwin-arm64")  UV_ASSET="uv-aarch64-apple-darwin.tar.gz" ;;
    "Darwin-x86_64") UV_ASSET="uv-x86_64-apple-darwin.tar.gz"  ;;
    "Linux-x86_64")  UV_ASSET="uv-x86_64-unknown-linux-gnu.tar.gz" ;;
    "Linux-aarch64") UV_ASSET="uv-aarch64-unknown-linux-gnu.tar.gz" ;;
    *)
        echo ""
        echo "ERROR: unrecognized system ($OS-$ARCH)."
        echo "Please contact your instructor."
        echo ""
        read -r -p "Press Enter to quit..." _
        exit 1
        ;;
esac

# --- Step 1: download uv ---
if [ ! -f "uv/uv" ]; then
    echo ""
    echo "[1/3] Downloading uv..."
    mkdir -p uv
    URL="https://github.com/astral-sh/uv/releases/latest/download/$UV_ASSET"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$URL" -o "uv/uv.tar.gz" || {
            echo "Download error. Check your internet connection."
            read -r -p "Press Enter to quit..." _
            exit 1
        }
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$URL" -O "uv/uv.tar.gz" || {
            echo "Download error."
            read -r -p "Press Enter to quit..." _
            exit 1
        }
    else
        echo "ERROR: neither curl nor wget is installed."
        read -r -p "Press Enter to quit..." _
        exit 1
    fi
    tar -xzf uv/uv.tar.gz -C uv --strip-components=1
    rm uv/uv.tar.gz
    chmod +x uv/uv
    echo "  uv downloaded."
else
    echo "[1/3] uv already present, skipping."
fi

# --- Step 2: create the Python environment ---
echo ""
echo "[2/3] Creating isolated Python environment..."
./uv/uv venv .venv --python 3.12 || {
    echo "ERROR while creating the Python environment."
    read -r -p "Press Enter to quit..." _
    exit 1
}
echo "  Environment created."

# --- Step 3: install dependencies ---
echo ""
echo "[3/3] Installing scientific libraries..."
echo "  (This may take a few minutes, please be patient.)"
./uv/uv pip install --python .venv/bin/python -r code/requirements.txt || {
    echo "ERROR while installing libraries."
    read -r -p "Press Enter to quit..." _
    exit 1
}

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo ""
echo "  You can now close this window and start MTA by"
echo "  double-clicking on:"
echo ""
echo "      start_MTA.bat       (Windows)"
echo "      start_MTA.command   (Mac)"
echo "      start_MTA.sh        (Linux)"
echo ""
read -r -p "  Press Enter to close..." _
