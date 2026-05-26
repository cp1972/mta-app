#!/bin/bash
# ============================================================
#  MTA for Master — Launcher (Mac/Linux)
# ============================================================
#  Lets the user pick between:
#    1. The Streamlit web app (default)
#    2. The MTA_v3.py interactive CLI
#    3. Batch / scripting help
#  All paths go through .venv/bin/python so that bundled
#  libraries (pandas, scikit-learn, etc.) are found.
# ============================================================

cd "$(dirname "$0")" || exit 1

# Verify installation
if [ ! -f ".venv/bin/python" ]; then
    echo ""
    echo "ERROR: MTA is not installed."
    echo "Please first double-click on:"
    echo "    install_first_run.command  (Mac)"
    echo "    install_first_run.sh       (Linux)"
    echo ""
    read -r -p "Press Enter to quit..." _
    exit 1
fi

show_menu() {
    clear 2>/dev/null || printf '\033c'
    echo ""
    echo "============================================================"
    echo "  MTA — Multi-Text Analyser"
    echo "============================================================"
    echo ""
    echo "  Choose an interface:"
    echo ""
    echo "    [1] Streamlit web app                   (default)"
    echo "        Recommended for beginners. Click-and-drag"
    echo "        interface in your browser."
    echo ""
    echo "    [2] Command-line interactive menu"
    echo "        Text-based menu, runs in this terminal."
    echo "        Same workflow as the original MTA.py."
    echo ""
    echo "    [3] Show batch / scripting usage"
    echo "        For Stata, R, or shell pipelines."
    echo ""
    echo "    [Q] Quit"
    echo ""
}

start_streamlit() {
    echo ""
    echo "============================================================"
    echo "  Starting Streamlit web app..."
    echo "============================================================"
    echo ""
    echo "  Your browser will open automatically."
    echo "  If nothing happens, open your browser at:"
    echo "      http://localhost:8501"
    echo ""
    echo "  IMPORTANT: DO NOT CLOSE this terminal while using MTA."
    echo "  Press Ctrl+C or close the window to stop MTA."
    echo "============================================================"
    echo ""
    (
        sleep 4
        if command -v open >/dev/null 2>&1; then     # Mac
            open http://localhost:8501
        elif command -v xdg-open >/dev/null 2>&1; then  # Linux
            xdg-open http://localhost:8501
        fi
    ) &
    ./.venv/bin/streamlit run code/streamlit_app.py \
        --server.headless true \
        --browser.gatherUsageStats false
}

start_cli() {
    echo ""
    echo "============================================================"
    echo "  Starting CLI interactive menu..."
    echo "============================================================"
    echo ""
    ./.venv/bin/python code/MTA_v3.py
    echo ""
    echo "CLI session ended."
    read -r -p "Press Enter to return to the main menu..." _
}

show_batch_help() {
    echo ""
    echo "============================================================"
    echo "  MTA_v3.py — Batch / scripting usage"
    echo "============================================================"
    echo ""
    ./.venv/bin/python code/MTA_v3.py --help
    echo ""
    echo "============================================================"
    echo "  Example for Stata / R / shell:"
    echo ""
    echo "  ./.venv/bin/python code/MTA_v3.py \\"
    echo "      --corpus  /path/to/corpus \\"
    echo "      --stopwords /path/to/stop.txt \\"
    echo "      --action nmf --n-topics 5 --json"
    echo "============================================================"
    echo ""
    read -r -p "Press Enter to return to the main menu..." _
}

# Main loop
while true; do
    show_menu
    read -r -p "Your choice [1]: " choice
    case "${choice:-1}" in
        1) start_streamlit; break ;;     # Streamlit blocks until closed
        2) start_cli ;;
        3) show_batch_help ;;
        [Qq]) echo ""; echo "Goodbye!"; exit 0 ;;
        *) echo ""; echo "  Invalid choice: $choice"; sleep 2 ;;
    esac
done
