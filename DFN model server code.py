"""
Flask web server for DFN Battery Discharge Simulator.
Run: python3 app.py
Then open: http://localhost:5000
"""

import importlib.util
import os
import io
import csv
import threading

import numpy as np
from flask import Flask, render_template, request, jsonify, send_file

# ── Load DFN simulation module (filename has spaces, use importlib) ──────────
_SIM_PATH = os.path.join(os.path.dirname(__file__), "DFN model simulation code.py")
_spec = importlib.util.spec_from_file_location("dfn_sim", _SIM_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_dfn_simulation          = _mod.run_dfn_simulation
export_csv                  = _mod.export_csv
recipe_to_electrode_props   = _mod.recipe_to_electrode_props
recipe_to_electrode_props_bk = _mod.recipe_to_electrode_props_bk
calc_film_resistance        = _mod.calc_film_resistance
calc_ca_tortuosity          = _mod.calc_ca_tortuosity
calc_ca_coverage            = _mod.calc_ca_coverage
load_material_params        = _mod.load_material_params
CA_PARAMS                   = _mod.CA_PARAMS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

# Store last simulation result for download endpoints
_last_result = {}
_sim_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ca_list")
def ca_list():
    """Return available CA materials and their parameters."""
    return jsonify({k: {
        "name":    v["name"],
        "sigma":   v["sigma"],
        "rho":     v["rho"],
        "phi_c":   v["phi_c"],
        "t":       v["t"],
        "geometry": v["geometry"],
    } for k, v in CA_PARAMS.items()})


@app.route("/api/perc_preview", methods=["POST"])
def perc_preview():
    """Real-time B-K percolation preview without running DFN simulation."""
    data     = request.get_json()
    material = data.get("material", "NCM")
    wt_CA    = float(data.get("wt_ca", 2.0))
    wt_binder= float(data.get("wt_binder", 2.0))
    ca1_type = data.get("ca1_type", "SuperP")
    ca2_type = data.get("ca2_type", None)
    ca2_ratio= float(data.get("ca2_ratio", 0.0))

    try:
        mat = load_material_params(material)
        wt_AM  = max(0.0, 100.0 - wt_CA - wt_binder)
        wt_CA1 = wt_CA * (1.0 - ca2_ratio)
        wt_CA2 = wt_CA * ca2_ratio
        use_ca2 = (ca2_type and ca2_type != ca1_type and wt_CA2 > 0)
        props = recipe_to_electrode_props_bk(
            wt_AM, wt_CA1, wt_CA2 if use_ca2 else 0.0, wt_binder,
            rho_AM=mat["rho_AM"],
            porosity=mat.get("eps_e_p", 0.335),
            R_p=mat["R_p_p"],
            ca1_type=ca1_type,
            ca2_type=ca2_type if use_ca2 else None,
        )
        R_film = calc_film_resistance(
            ca1_type, props["phi_CA1"],
            ca2_type if use_ca2 else None,
            props["phi_CA2"] if use_ca2 else 0.0,
        )
        tau_extra = calc_ca_tortuosity(
            ca1_type, props["phi_CA1"],
            ca2_type if use_ca2 else None,
            props["phi_CA2"] if use_ca2 else 0.0,
        )
        theta_ca = calc_ca_coverage(
            ca1_type, props["phi_CA1"],
            ca2_type if use_ca2 else None,
            props["phi_CA2"] if use_ca2 else 0.0,
        )
        return jsonify({
            "sigma_eff":  round(props["sigma_eff"], 4),
            "eps_s":      round(props["eps_s"], 4),
            "eps_e":      round(props["eps_e"], 4),
            "a_p":        round(props["a"], 0),
            "perc_index": round(props["perc_index"], 3),
            "phi_CA1":    round(props["phi_CA1"], 4),
            "phi_CA2":    round(props["phi_CA2"], 4),
            "phi_c1_eff": round(props["phi_c1_eff"], 4),
            "phi_c2_eff": round(props["phi_c2_eff"] if props["phi_c2_eff"] is not None else 0, 4),
            "R_film":     round(R_film, 6),
            "tau_extra":  round(tau_extra, 4),
            "theta_ca":   round(theta_ca, 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """Run DFN simulation and return Plotly JSON + summary stats."""
    data = request.get_json()

    material    = data.get("material", "NCM")
    C_rate      = float(data.get("crate", 1.0))
    wt_CA       = float(data.get("wt_ca", 0))
    wt_binder   = float(data.get("wt_binder", 0))
    use_recipe  = data.get("use_recipe", False)
    ca1_type    = data.get("ca1_type", "SuperP")
    ca2_type    = data.get("ca2_type", None)
    ca2_ratio   = float(data.get("ca2_ratio", 0.0))
    tau_elyte_p = data.get("tau_elyte_p", None)   # optional direct τ override
    if tau_elyte_p is not None:
        tau_elyte_p = float(tau_elyte_p)

    try:
        t, V, p, geo = run_dfn_simulation(
            C_rate=C_rate,
            material=material,
            recipe_CA=wt_CA if use_recipe else None,
            recipe_Binder=wt_binder if use_recipe else None,
            ca1_type=ca1_type,
            ca2_type=ca2_type,
            ca2_ratio=ca2_ratio,
            tau_elyte_p=tau_elyte_p,
            N_n=15, N_s=8, N_p=15, N_r=8,
            plot=False,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    Q_cell = p["Q_cell"]
    cap_Ah = t * Q_cell * C_rate / 3600.0
    mass_AM_g = p["eps_s_p"] * p["L_p"] * p["A_cell"] * p["rho_AM"] * 1e6
    cap_mAhg = cap_Ah * 1000.0 / mass_AM_g
    V_list   = V.tolist()
    t_list   = t.tolist()
    cap_list = cap_mAhg.tolist()

    # Percolation conductivity preview (Bauhofer-Kovacs model)
    sigma_info = None
    if use_recipe and wt_CA > 0 and wt_binder > 0:
        mat = load_material_params(material)
        wt_AM_total = 100.0 - wt_CA - wt_binder
        # Split CA into CA1/CA2 by ca2_ratio
        wt_CA1 = wt_CA * (1.0 - ca2_ratio)
        wt_CA2 = wt_CA * ca2_ratio
        use_ca2 = (ca2_type and ca2_type != ca1_type and wt_CA2 > 0)
        props = recipe_to_electrode_props_bk(
            wt_AM_total, wt_CA1, wt_CA2 if use_ca2 else 0.0, wt_binder,
            rho_AM=mat["rho_AM"],
            porosity=p["eps_e_p"],
            R_p=p["R_p_p"],
            ca1_type=ca1_type,
            ca2_type=ca2_type if use_ca2 else None,
        )
        R_film = calc_film_resistance(
            ca1_type, props["phi_CA1"],
            ca2_type if use_ca2 else None,
            props["phi_CA2"] if use_ca2 else 0.0,
        )
        tau_extra = calc_ca_tortuosity(
            ca1_type, props["phi_CA1"],
            ca2_type if use_ca2 else None,
            props["phi_CA2"] if use_ca2 else 0.0,
        )
        theta_ca = calc_ca_coverage(
            ca1_type, props["phi_CA1"],
            ca2_type if use_ca2 else None,
            props["phi_CA2"] if use_ca2 else 0.0,
        )
        sigma_info = {
            "sigma_eff":  round(props["sigma_eff"], 4),
            "eps_s":      round(props["eps_s"], 4),
            "eps_e":      round(props["eps_e"], 4),
            "a_p":        round(props["a"], 0),
            "perc_index": round(props["perc_index"], 3),
            "phi_CA1":    round(props["phi_CA1"], 4),
            "phi_CA2":    round(props["phi_CA2"], 4),
            "phi_c1_eff": round(props["phi_c1_eff"], 4),
            "phi_c2_eff": round(props["phi_c2_eff"] if props["phi_c2_eff"] is not None else 0, 4),
            "ca1_type":   ca1_type,
            "ca2_type":   ca2_type if use_ca2 else None,
            "R_film":     round(R_film, 6),
            "tau_extra":  round(tau_extra, 4),
            "theta_ca":   round(theta_ca, 4),
        }

    summary = {
        "V_initial": round(float(V[0]), 4),
        "V_final":   round(float(V[-1]), 4),
        "capacity":  round(float(cap_mAhg[-1]), 2),
        "duration_h": round(float(t[-1]) / 3600, 3),
        "n_steps":   len(t),
    }

    # Store for download
    with _sim_lock:
        _last_result["t"]       = t_list
        _last_result["V"]       = V_list
        _last_result["cap"]     = cap_list
        _last_result["p"]       = {"Q_cell": Q_cell, "V_min": p["V_min"], "V_max": p["V_max"]}
        _last_result["material"] = material
        _last_result["crate"]   = C_rate

    return jsonify({
        "t":         t_list,
        "V":         V_list,
        "cap":       cap_list,
        "summary":   summary,
        "sigma_info": sigma_info,
    })


@app.route("/api/download/csv")
def download_csv():
    """Return last simulation result as CSV file."""
    with _sim_lock:
        if not _last_result:
            return jsonify({"error": "No simulation run yet"}), 404
        t   = _last_result["t"]
        V   = _last_result["V"]
        cap = _last_result["cap"]
        mat = _last_result["material"]
        cr  = _last_result["crate"]
        Q   = _last_result["p"]["Q_cell"]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["time_s", "time_h", "voltage_V", "capacity_Ah", "SOC",
                     "material", "C_rate"])
    for i in range(len(t)):
        soc = max(0.0, 1.0 - (Q * cr * t[i]) / (Q * 3600.0))
        writer.writerow([
            f"{t[i]:.4f}",
            f"{t[i]/3600:.6f}",
            f"{V[i]:.6f}",
            f"{cap[i]:.6f}",
            f"{soc:.6f}",
            mat,
            f"{cr:.2f}",
        ])

    buf.seek(0)
    filename = f"DFN_{mat}_{cr:.1f}C.csv"
    return send_file(
        io.BytesIO(buf.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/api/download/png")
def download_png():
    """Return last simulation result as PNG chart."""
    with _sim_lock:
        if not _last_result:
            return jsonify({"error": "No simulation run yet"}), 404
        t   = np.array(_last_result["t"])
        V   = np.array(_last_result["V"])
        cap = np.array(_last_result["cap"])
        mat = _last_result["material"]
        cr  = _last_result["crate"]
        V_min = _last_result["p"]["V_min"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"DFN Discharge — {mat} | {cr}C", fontsize=13, fontweight="bold")

    axes[0].plot(t / 3600, V, "b-", lw=2)
    axes[0].axhline(V_min, color="r", ls="--", alpha=0.5, label=f"V_min={V_min}V")
    axes[0].set_xlabel("Time [h]")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].set_title("Voltage vs Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cap, V, "g-", lw=2)
    axes[1].axhline(V_min, color="r", ls="--", alpha=0.5)
    axes[1].set_xlabel("Specific Capacity [mAh/g]")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_title("Voltage vs Specific Capacity")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    filename = f"DFN_{mat}_{cr:.1f}C.png"
    return send_file(buf, mimetype="image/png",
                     as_attachment=True, download_name=filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 60)
    print("  DFN Battery Simulator — Web Interface")
    print(f"  http://localhost:{port}")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=port, threaded=False)
