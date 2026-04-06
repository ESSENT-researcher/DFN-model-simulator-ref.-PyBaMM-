import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv, os
from DFN_model_simulation_code import run_dfn_simulation

CSV_FILES = {
    "0.25C": "lfp_025C.csv",
    "0.5C":  "lfp_05C.csv",
    "1C":    "lfp_1C.csv",
    "2C":    "lfp_2C.csv",
    "4C":    "lfp_4C.csv",
}
COLORS = {"0.25C":"#9467bd","0.5C":"#1f77b4","1C":"#2ca02c","2C":"#d62728","4C":"#ff7f0e"}

def load_csv(filepath):
    data = []
    with open(filepath, newline="") as f:
        for row in csv.reader(f):
            try:
                data.append([float(row[0]), float(row[1])])
            except:
                continue
    arr = np.array(data)
    arr = arr[arr[:,0].argsort()]
    arr[:,0] = np.clip(arr[:,0], 0, None)
    return arr

def ah_to_mAhg(ah, p):
    mass = p["eps_s_p"] * p["L_p"] * p["A_cell"] * p["rho_AM"] * 1e6
    return ah * 1000.0 / mass

def run_sim(c_rate):
    print(f"  시뮬레이션: LFP {c_rate}C ...")
    t, V, p, geo = run_dfn_simulation(
        C_rate=c_rate, material="LFP",
        N_n=15, N_s=8, N_p=15, N_r=8,
        plot=False, cell_type="Prada2013"
    )
    cap_Ah = t * p["Q_cell"] * c_rate / 3600.0
    cap_mAhg = ah_to_mAhg(cap_Ah, p)
    return cap_Ah, cap_mAhg, V, p

def get_metrics(sim_Ah, sim_V, exp_Ah, exp_V, Vmin=2.0):
    mask = exp_V >= Vmin
    if mask.sum() < 2:
        return None
    f = interp1d(sim_Ah, sim_V, bounds_error=False, fill_value=np.nan)
    si = f(exp_Ah[mask])
    ok = ~np.isnan(si)
    if ok.sum() < 2:
        return None
    r = si[ok] - exp_V[mask][ok]
    return {"RMSE": np.sqrt(np.mean(r**2))*1000, "MAE": np.mean(np.abs(r))*1000, "dCap": sim_Ah[-1]-exp_Ah[mask][-1]}

print("=" * 50)
print("  DFN 검증 - LFP vs. Prada 2012 (Prada2013 셀)")
print("=" * 50)

exp, sims, mets = {}, {}, {}

for cr, fp in CSV_FILES.items():
    if os.path.exists(fp):
        exp[cr] = load_csv(fp)
        print(f"  로드: {fp} ({len(exp[cr])}점)")
    else:
        print(f"  파일 없음: {fp}")

for cr, arr in exp.items():
    c = float(cr.replace("C", ""))
    try:
        cAh, cmAhg, V, p = run_sim(c)
        sims[cr] = (cAh, cmAhg, V, p)
        m = get_metrics(cAh, V, arr[:,0], arr[:,1], p["V_min"])
        if m:
            mets[cr] = m
            print(f"  [{cr}] RMSE={m['RMSE']:.1f}mV  MAE={m['MAE']:.1f}mV  dCap={m['dCap']:+.3f}Ah")
    except Exception as e:
        print(f"  [{cr}] 실패: {e}")

fig, ax = plt.subplots(figsize=(10, 7))
for cr, arr in exp.items():
    col = COLORS.get(cr, "gray")
    if cr in sims:
        cAh, cmAhg, V, p = sims[cr]
        emg = ah_to_mAhg(arr[:,0], p)
        ax.plot(cmAhg, V, "-", lw=2.2, color=col, label=f"Sim {cr}")
    else:
        emg = arr[:,0] * 1000 / 2.0
    ax.plot(emg, arr[:,1], "--", lw=1.6, alpha=0.9, marker="o", ms=3.5, markevery=4, color=col, label=f"Exp {cr}")

from matplotlib.lines import Line2D
handles, _ = ax.get_legend_handles_labels()
ax.legend(handles=handles+[Line2D([0],[0],color="k",lw=2,ls="-",label="Simulation (DFN)"),
          Line2D([0],[0],color="k",lw=1.6,ls="--",marker="o",ms=4,label="Exp (Prada 2012)")],
          fontsize=9, ncol=2, loc="lower left")
ax.set_xlabel("Specific Capacity [mAh/g]", fontsize=12)
ax.set_ylabel("Voltage [V]", fontsize=12)
ax.set_title("LFP/Graphite - DFN Simulation vs. Prada 2012 (Prada2013 Cell)", fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.25, ls=":")
if mets:
    lines = ["C-rate   RMSE    MAE    dCap", "-"*34]
    for cr, m in mets.items():
        lines.append(f"{cr:<7}  {m['RMSE']:>5.1f}mV {m['MAE']:>5.1f}mV {m['dCap']:>+6.3f}Ah")
    ax.text(0.02, 0.04, "\n".join(lines), transform=ax.transAxes, fontsize=8,
            fontfamily="monospace", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88, edgecolor="gray"))
plt.tight_layout()
plt.savefig("validation_LFP.png", dpi=150)
print("  저장완료: validation_LFP.png")
plt.show()
print("\n===결과===")
for cr, m in mets.items():
    print(f"  {cr}: RMSE={m['RMSE']:.1f}mV  MAE={m['MAE']:.1f}mV  dCap={m['dCap']:+.3f}Ah")
