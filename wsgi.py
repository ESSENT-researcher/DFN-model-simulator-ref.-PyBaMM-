"""
Gunicorn entry point for Render deployment.
"""
import importlib.util
import os

_SIM_PATH = os.path.join(os.path.dirname(__file__), "DFN model server code.py")
_spec = importlib.util.spec_from_file_location("dfn_model_simulator", _SIM_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

app = _mod.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
