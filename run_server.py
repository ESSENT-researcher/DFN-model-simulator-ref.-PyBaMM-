"""
DFN Model Simulator — Server Launcher
Starts the Flask server and opens the web interface in the default browser.
Usage: python3 run_server.py
"""

import importlib.util
import os
import threading
import webbrowser
import time

HOST = "localhost"
PORT = 5001
URL  = f"http://{HOST}:{PORT}"

# ── Load DFN model simulator module ──────────────────────────────────────────
_SIM_PATH = os.path.join(os.path.dirname(__file__), "DFN model server code.py")
_spec = importlib.util.spec_from_file_location("dfn_model_simulator", _SIM_PATH)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

app = _mod.app


def open_browser():
    """Wait briefly for the server to start, then open the browser."""
    time.sleep(1.5)
    webbrowser.open(URL)


if __name__ == "__main__":
    print("=" * 60)
    print("  DFN Model Simulator — Web Interface")
    print(f"  Opening browser at {URL}")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60)

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(debug=False, host=HOST, port=PORT, threaded=False)
