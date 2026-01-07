#!/usr/bin/env bash
# Simple launcher for chatbot GUI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


# Ollama check removed (Local Inference Mode)


# Launch chatbot GUI
# Check for virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

# Check if --debug flag is present
DEBUG_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--debug" ]; then
        DEBUG_MODE=true
        break
    fi
done

# Launch chatbot GUI (suppress verbose stdout unless debug, but ALWAYS show errors)
if [ "$DEBUG_MODE" = true ]; then
    "$PYTHON_CMD" "$SCRIPT_DIR/run_chatbot.py" "$@"
else
    # Only suppress stdout, keep stderr visible for errors
    "$PYTHON_CMD" "$SCRIPT_DIR/run_chatbot.py" "$@" 2>&1 | grep -v "^DEBUG:" || true
fi

