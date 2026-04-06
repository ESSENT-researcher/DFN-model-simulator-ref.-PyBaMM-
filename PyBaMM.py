"""
DFN Discharge Simulation — Dose 2020 비교용
PyBaMM Chen2020 파라미터셋 / C/2 방전 / 2.5~4.2 V

실행 후 출력된 JSON을 Claude에게 붙여넣으세요.
"""

import pybamm
import json
import numpy as np

# ── 모델 & 파라미터 ──────────────────────────────────────────
model = pybamm.lithium_ion.DFN()
param = pybamm.ParameterValues("Chen2020")

C_rate = 0.5   # Dose 2020 실험 조건: C/2
param["Current function [A]"] = param["Nominal cell capacity [A.h]"] * C_rate

# ── 시뮬레이션 (Experiment 없이 단순 방전) ──────────────────
sim = pybamm.Simulation(model, parameter_values=param)
t_end = 1 / C_rate * 3600 * 1.05   # 여유 5% 추가
sim.solve([0, t_end])

sol = sim.solution
voltage     = sol["Terminal voltage [V]"].entries
time_s      = sol["Time [s]"].entries

# ── mAh/g 변환 (양극 활물질 질량 기준) ──────────────────────
Q_nom   = float(param["Nominal cell capacity [A.h]"])
L_p     = float(param["Positive electrode thickness [m]"])
eps_p   = float(param["Positive electrode active material volume fraction"])
H       = float(param["Electrode height [m]"])
W       = float(param["Electrode width [m]"])
rho_p   = 3262.0                               # kg/m³ (NMC)
mass_p_g = L_p * eps_p * H * W * rho_p * 1000 # g

capacity_mAh_g = (time_s * C_rate * Q_nom / 3600) * 1000 / mass_p_g

# ── 2.5 V 컷오프 적용 ────────────────────────────────────────
mask   = voltage >= 2.5
v_out  = voltage[mask]
q_out  = capacity_mAh_g[mask]

# ── JSON 출력 ────────────────────────────────────────────────
data = {
    "voltage":          [round(float(v), 4) for v in v_out],
    "capacity_mAh_g":   [round(float(q), 3) for q in q_out],
    "Q_nom_Ah":         round(Q_nom, 4),
    "mass_cathode_g":   round(mass_p_g, 4),
    "C_rate":           C_rate,
}

print(json.dumps(data))

# 기존 코드 맨 아래에 추가
with open("dfn_output.json", "w") as f:
    json.dump(data, f)
print("저장 완료: dfn_output.json")