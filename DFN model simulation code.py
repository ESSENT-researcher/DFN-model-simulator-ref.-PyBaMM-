"""
=============================================================================
DFN (Doyle-Fuller-Newman) Model — PyBaMM-Faithful Implementation
=============================================================================
Replicates PyBaMM's DFN model numerics from scratch:
  - 5 governing equations (solid diffusion, electrolyte diffusion,
    solid potential, electrolyte potential, Butler-Volmer kinetics)
  - Finite Volume Method with proper gradient/divergence matrices
  - Spherical FVM for particle-internal diffusion
  - Pseudo-transient DAE approach via scipy solve_ivp (BDF)
  - Chen2020 parameter set (LG M50, NCM811/Graphite, 5Ah)

Author: Capstone Design Project
=============================================================================
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy import sparse
import matplotlib.pyplot as plt
import json
import time as timer

# =============================================================================
# 1. CONSTANTS
# =============================================================================
F = 96485.33212  # Faraday constant [C/mol]
R_gas = 8.314462  # Gas constant [J/(mol·K)]

# =============================================================================
# 2. FINITE VOLUME METHOD INFRASTRUCTURE
# =============================================================================

def build_uniform_mesh(L, N):
    """Build 1D uniform FVM mesh.
    Returns dict with edges, nodes, d_edges, d_nodes.
    """
    edges = np.linspace(0, L, N + 1)
    nodes = 0.5 * (edges[1:] + edges[:-1])
    d_edges = np.diff(edges)  # cell widths (N,)
    d_nodes = np.diff(nodes)  # node spacings (N-1,)
    return {"edges": edges, "nodes": nodes, "d_edges": d_edges, "d_nodes": d_nodes,
            "N": N, "L": L}


def build_gradient_matrix(mesh):
    """Gradient matrix: (N-1) x N, maps node values to edge gradients.
    G @ phi = [  (phi[1]-phi[0])/d_nodes[0],
                 (phi[2]-phi[1])/d_nodes[1], ... ]
    """
    N = mesh["N"]
    d = mesh["d_nodes"]  # (N-1,)
    diag_m = -1.0 / d
    diag_p = 1.0 / d
    G = sparse.diags([diag_m, diag_p], [0, 1], shape=(N - 1, N), format="csr")
    return G


def build_divergence_matrix_cartesian(mesh):
    """Divergence matrix for Cartesian coords: N x (N+1).
    Maps edge fluxes (including boundary edges) to node divergences.
    D @ flux_edges = [ (flux[1]-flux[0])/dx[0], (flux[2]-flux[1])/dx[1], ... ]
    """
    N = mesh["N"]
    d = mesh["d_edges"]  # (N,)
    diag_m = -1.0 / d
    diag_p = 1.0 / d
    D = sparse.diags([diag_m, diag_p], [0, 1], shape=(N, N + 1), format="csr")
    return D


def build_spherical_fvm(R_p, N_r):
    """Build FVM operators for spherical particle diffusion.

    PyBaMM approach:
    - Nodes at cell centers in r-direction [0, R_p]
    - Gradient: (N_r-1) x N_r
    - Divergence: accounts for r^2 geometry
      div(N) = (1/V_cell) * [r_right^2 * N_right - r_left^2 * N_left]
      where V_cell = (r_right^3 - r_left^3)/3

    Returns dict with mesh info and operators.
    """
    edges = np.linspace(0, R_p, N_r + 1)
    nodes = 0.5 * (edges[1:] + edges[:-1])
    d_nodes = np.diff(nodes)  # (N_r - 1,)
    d_edges = np.diff(edges)  # (N_r,)

    # Gradient matrix: (N_r-1) x N_r
    inv_d = 1.0 / d_nodes
    G = sparse.diags([-inv_d, inv_d], [0, 1], shape=(N_r - 1, N_r), format="csr")

    # Divergence matrix for spherical coords: N_r x (N_r+1)
    # Cell volumes: V_i = (r_right^3 - r_left^3) / 3
    r_left = edges[:-1]
    r_right = edges[1:]
    vol = (r_right**3 - r_left**3) / 3.0
    inv_vol = 1.0 / vol

    # r^2 at edges (for flux weighting)
    r2_edges = edges**2  # (N_r+1,)

    # D[i,i] = -r2_edges[i] / vol[i], D[i,i+1] = r2_edges[i+1] / vol[i]
    diag_m = -r2_edges[:-1] * inv_vol
    diag_p = r2_edges[1:] * inv_vol
    D = sparse.diags([diag_m, diag_p], [0, 1], shape=(N_r, N_r + 1), format="csr")

    return {
        "edges": edges, "nodes": nodes, "d_edges": d_edges, "d_nodes": d_nodes,
        "N": N_r, "R_p": R_p,
        "G": G, "D": D,
        "r2_edges": r2_edges, "vol": vol
    }


def apply_neumann_bc(internal_fluxes, left_flux, right_flux):
    """Prepend left boundary flux and append right boundary flux to internal fluxes.
    internal_fluxes: (N-1,) from gradient matrix
    Returns: (N+1,) edge fluxes including boundaries.
    """
    return np.concatenate(([left_flux], internal_fluxes, [right_flux]))


# =============================================================================
# 3. ELECTROCHEMICAL PARAMETERS (Chen2020 — LG M50)
# =============================================================================

def load_material_params(material="NCM"):
    """Load cathode-specific parameters for a given material type.
    Returns dict with cathode parameters and OCP function.
    Supported: 'NCM' (NCM811), 'LFP' (LiFePO4), 'LMFP' (LiMn0.6Fe0.4PO4),
               'LMFP64' (Mn0.6Fe0.4), 'LMFP73' (Mn0.7Fe0.3), 'LMFP82' (Mn0.8Fe0.2)
    """
    if material == "NCM":
        return {
            "c_s_max_p": 63104.0,
            "c_s_init_p": 17038.0,    # sto_p_init = 0.27
            "D_s_p": 4.0e-15,
            "R_p_p": 5.22e-6,
            "sigma_p": 0.18,
            "k_p": 3.42e-6,
            "rho_AM": 4.80,            # g/cm³
            "OCP_p": OCP_nmc_Chen2020,
            "V_min": 2.5,
            "V_max": 4.2,
            "sto_p_init": 0.27,
            "sto_p_full": 0.90,        # approx. fully discharged
        }
    elif material == "LFP":
        return {
            "c_s_max_p": 22806.0,
            "c_s_init_p": 22806.0 * 0.05,  # sto_p_init = 0.05 (charged)
            "D_s_p": 5.9e-18,
            "R_p_p": 5.0e-8,           # 50nm nanoparticles
            "sigma_p": 0.338,
            "k_p": 6.0e-7,
            "rho_AM": 3.60,
            "OCP_p": OCP_LFP_Prada2013,
            "V_min": 2.5,
            "V_max": 3.6,
            "sto_p_init": 0.05,
            "sto_p_full": 0.95,
        }
    elif material == "LMFP":
        return {
            "c_s_max_p": 22900.0,
            "c_s_init_p": 22900.0 * 0.05,  # sto_p_init = 0.05 (charged)
            "D_s_p": 1.0e-16,          # carbon-coated LMFP, enhanced
            "R_p_p": 1.0e-6,           # 1um (carbon-coated secondary particles)
            "sigma_p": 0.5,            # carbon-coated composite
            "k_p": 1.0e-6,
            "rho_AM": 3.50,
            "OCP_p": OCP_LMFP,
            "V_min": 2.5,
            "V_max": 4.4,
            "sto_p_init": 0.05,
            "sto_p_full": 0.95,
        }
    elif material == "LMFP64":
        return {
            "c_s_max_p": 22800.0,          # LiMn0.6Fe0.4PO4 (pybamm_all_parameters.json)
            "c_s_init_p": 22800.0 * 0.05,
            "D_s_p": 1.0e-15,
            "R_p_p": 1.0e-7,               # 100nm (carbon-coated nanoparticles)
            "sigma_p": 0.005,
            "k_p": 1.0e-6,
            "rho_AM": 3.48,
            "OCP_p": OCP_LMFP64,
            "V_min": 2.5,
            "V_max": 4.4,
            "sto_p_init": 0.05,
            "sto_p_full": 0.95,
        }
    elif material == "LMFP73":
        return {
            "c_s_max_p": 22850.0,          # LiMn0.7Fe0.3PO4
            "c_s_init_p": 22850.0 * 0.05,
            "D_s_p": 8.0e-16,
            "R_p_p": 1.0e-7,
            "sigma_p": 0.001,
            "k_p": 1.0e-6,
            "rho_AM": 3.465,
            "OCP_p": OCP_LMFP73,
            "V_min": 2.5,
            "V_max": 4.4,
            "sto_p_init": 0.05,
            "sto_p_full": 0.95,
        }
    elif material == "LMFP82":
        return {
            "c_s_max_p": 22900.0,          # LiMn0.8Fe0.2PO4
            "c_s_init_p": 22900.0 * 0.05,
            "D_s_p": 5.0e-16,
            "R_p_p": 1.0e-7,
            "sigma_p": 0.0001,
            "k_p": 1.0e-6,
            "rho_AM": 3.45,
            "OCP_p": OCP_LMFP82,
            "V_min": 2.5,
            "V_max": 4.4,
            "sto_p_init": 0.05,
            "sto_p_full": 0.95,
        }
    else:
        raise ValueError(f"Unknown material: {material}. Use 'NCM', 'LFP', 'LMFP', 'LMFP64', 'LMFP73', or 'LMFP82'.")


def D_s_LFP_Safari2011(c_s, c_s_max):
    """Concentration-dependent solid diffusivity for LFP (Safari & Delacourt 2011).
    D = D0 / (1 + y)^1.6,  where y = c_s / c_s_max (stoichiometry)
    D0 = 1.18e-18 m²/s (at fully delithiated state, y=0)
    Returns scalar (averaged over particle nodes).
    """
    D0 = 1.18e-18
    y = np.mean(np.clip(c_s / c_s_max, 0.0, 1.0))
    return D0 / (1.0 + y) ** 1.6


def load_prada2013_params():
    """All parameters from Safari & Delacourt 2011 (J. Electrochem. Soc. 158, A562)
    for commercial 2.3Ah 26650 Graphite/LiFePO4 cell (C26650).
    Table I (OCP functions) + Table II (model parameters at 25°C).
    """
    p = {}
    p["T"] = 298.15
    # Electrode area from paper: A_c=1694 cm², A_a=1755 cm²; use LFP electrode area
    p["A_cell"] = 0.1694   # m²
    p["H"] = p["A_cell"] ** 0.5
    p["W"] = p["A_cell"] ** 0.5

    # Electrode thicknesses (Table II)
    p["L_n"] = 3.4e-5      # 34 µm (graphite, measured from SEM)
    p["L_s"] = 2.5e-5      # 25 µm (separator)
    p["L_p"] = 7.0e-5      # 70 µm (LFP, Table II)

    # Porosity — kept from PyBaMM Prada2013
    p["eps_e_n"] = 0.36
    p["eps_e_s"] = 0.45
    p["eps_e_p"] = 0.426

    # Active material volume fraction (Table II)
    p["eps_s_n"] = 0.555   # 0.55~0.56 midpoint
    p["eps_s_p"] = 0.432   # 0.428~0.435 midpoint

    # Particle radii (Table II)
    p["R_p_n"] = 3.5e-6    # 3.5 µm graphite
    p["R_p_p"] = 3.65e-8   # 36.5 nm LFP nanoparticles

    # Solid diffusivities (Table II)
    p["D_s_n"] = 2.0e-14   # graphite, constant [m²/s]
    p["D_s_p"] = D_s_LFP_Safari2011  # LFP: concentration-dependent callable

    # Maximum solid concentration (Table II)
    p["c_s_max_n"] = 31370.0
    p["c_s_max_p"] = 22806.0

    # Initial stoichiometry: fully charged state (Hypothesis 2, Table II)
    # graphite x_max=0.80, LFP y_min=0.03
    p["c_s_init_n"] = 31370.0 * 0.80   # = 25096
    p["c_s_init_p"] = 22806.0 * 0.03   # = 684.18

    p["sigma_n"] = 215.0
    p["sigma_p"] = 0.33795074
    p["sigma_eff_n"] = p["sigma_n"]
    p["sigma_eff_p"] = p["sigma_p"]

    p["brugg_n"] = 1.5
    p["brugg_s"] = 1.5
    p["brugg_p"] = 1.5

    p["a_n"] = 3.0 * p["eps_s_n"] / p["R_p_n"]
    p["a_p"] = 3.0 * p["eps_s_p"] / p["R_p_p"]

    # Rate constants — paper units mol/[m²s(mol/m³)^1.5]; convert ×F to m^2.5/mol^0.5/s
    # k_a = 8.19e-12 × 96485 ≈ 7.9e-7; k_p from Kashkooli2017
    p["k_n"] = 7.9e-7
    p["k_p"] = 6.0e-7

    p["c_e_init"] = 1000.0  # 1 mol/l = 1000 mol/m³ (Table II)
    p["t_plus"] = 0.36
    p["therm_factor"] = 1.0

    p["Q_cell"] = 2.3
    p["V_min"] = 2.0
    p["V_max"] = 3.6

    p["OCP_n"] = OCP_graphite_Prada2013
    p["OCP_p"] = OCP_LFP_Prada2013

    p["kappa_e_func"] = electrolyte_conductivity_Prada2013
    p["D_e_func"] = electrolyte_diffusivity_Prada2013

    return p


def electrolyte_conductivity_Prada2013(c_e):
    """Electrolyte conductivity [S/m] for Prada2013 (LiPF6 in EC:EMC 3:7)."""
    c_e = np.clip(c_e, 1.0, 5000.0)
    c = c_e / 1e6  # mol/m³ -> mol/cm³ (Prada convention)
    sigma_e = (4.1253e-4 + 5.007*c - 4721.2*c**2 + 1.5094e6*c**3 - 1.6018e8*c**4) * 1e3
    return np.maximum(sigma_e, 1e-6)


def electrolyte_diffusivity_Prada2013(c_e):
    """Electrolyte diffusivity [m²/s] for Prada2013 — constant."""
    return np.full_like(np.asarray(c_e, dtype=float), 2.0e-10)


def load_chen2020_params():
    """All parameters from Chen2020 for LG M50 cell (NCM811/Graphite, 5Ah)."""
    p = {}

    # Temperature
    p["T"] = 298.15  # [K]

    # Cell geometry
    p["H"] = 0.065   # Electrode height [m]
    p["W"] = 1.58    # Electrode width [m]
    p["A_cell"] = p["H"] * p["W"]  # Cross-sectional area [m^2]

    # Thicknesses [m]
    p["L_n"] = 8.52e-5   # Negative electrode
    p["L_s"] = 1.2e-5    # Separator
    p["L_p"] = 7.56e-5   # Positive electrode

    # Porosity (electrolyte volume fraction)
    p["eps_e_n"] = 0.25
    p["eps_e_s"] = 0.47
    p["eps_e_p"] = 0.335

    # Active material volume fraction
    p["eps_s_n"] = 0.75
    p["eps_s_p"] = 0.665

    # Particle radii [m]
    p["R_p_n"] = 5.86e-6
    p["R_p_p"] = 5.22e-6

    # Solid-phase diffusivity [m^2/s]
    p["D_s_n"] = 3.3e-14
    p["D_s_p"] = 4.0e-15

    # Maximum concentration in solid [mol/m^3]
    p["c_s_max_n"] = 33133.0
    p["c_s_max_p"] = 63104.0

    # Initial concentration in solid [mol/m^3]
    p["c_s_init_n"] = 29866.0
    p["c_s_init_p"] = 17038.0

    # Electronic conductivity [S/m]
    p["sigma_n"] = 215.0
    p["sigma_p"] = 0.18  # Will be overridden by percolation if recipe given

    # Bruggeman exponents
    p["brugg_n"] = 1.5
    p["brugg_s"] = 1.5
    p["brugg_p"] = 1.5

    # Effective electronic conductivity (Bruggeman on electrode volume fraction)
    # PyBaMM Chen2020: electrode Bruggeman = 0 for electronic conductivity
    # (i.e., sigma_eff = sigma * eps_s^0 = sigma)
    # But for electrolyte: brugg = 1.5
    p["sigma_eff_n"] = p["sigma_n"]  # No Bruggeman for electronic (Chen2020)
    p["sigma_eff_p"] = p["sigma_p"]

    # Specific interfacial area [1/m]
    p["a_n"] = 3.0 * p["eps_s_n"] / p["R_p_n"]
    p["a_p"] = 3.0 * p["eps_s_p"] / p["R_p_p"]

    # Reaction rate constants [m^2.5 / (mol^0.5 · s)]
    # PyBaMM Chen2020: j0 = k * F * ce^0.5 * cs_surf^0.5 * (cs_max - cs_surf)^0.5
    # k_n = 6.48e-7, k_p = 3.42e-6 (from exchange current density functions)
    p["k_n"] = 6.48e-7
    p["k_p"] = 3.42e-6

    # Electrolyte
    p["c_e_init"] = 1000.0  # [mol/m^3]
    p["t_plus"] = 0.2594    # Transference number
    p["therm_factor"] = 1.0  # Thermodynamic factor

    # Nominal capacity
    p["Q_cell"] = 5.0  # [Ah]

    # Voltage cutoffs
    p["V_min"] = 2.5
    p["V_max"] = 4.2

    # OCP function references (can be overridden by material selection)
    p["OCP_n"] = OCP_graphite_Chen2020
    p["OCP_p"] = OCP_nmc_Chen2020

    # Electrolyte property functions
    p["kappa_e_func"] = electrolyte_conductivity
    p["D_e_func"] = electrolyte_diffusivity

    return p


# =============================================================================
# 4. OCP FUNCTIONS
# =============================================================================

def OCP_graphite_Chen2020(sto):
    """Open-circuit potential for Graphite negative electrode (Chen2020).
    sto = c_s_surf / c_s_max, clipped to (0, 1).
    """
    sto = np.clip(sto, 1e-6, 1.0 - 1e-6)
    U = (1.9793 * np.exp(-39.3631 * sto)
         + 0.2482
         - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
         - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
         - 0.0205 * np.tanh(30.4444 * (sto - 0.6103)))
    return U


def OCP_graphite_Prada2013(sto):
    """Open-circuit potential for Graphite negative electrode (Prada2013).
    U_a = 0.6379 + 0.5416*exp(-305.5309*x)
          + 0.044*tanh((x-0.1958)/0.1088)
          - 0.1978*tanh((x-1.0571)/0.0854)
          - 0.6875*tanh((x+0.0117)/0.0529)
          - 0.0175*tanh((x-0.5692)/0.0875)
    """
    x = np.clip(sto, 1e-6, 1.0 - 1e-6)
    U = (0.6379
         + 0.5416 * np.exp(-305.5309 * x)
         + 0.044  * np.tanh((x - 0.1958) / 0.1088)
         - 0.1978 * np.tanh((x - 1.0571) / 0.0854)
         - 0.6875 * np.tanh((x + 0.0117) / 0.0529)
         - 0.0175 * np.tanh((x - 0.5692) / 0.0875))
    return U


def OCP_nmc_Chen2020(sto):
    """Open-circuit potential for NMC811 positive electrode (Chen2020).
    sto = c_s_surf / c_s_max, clipped to (0, 1).
    """
    sto = np.clip(sto, 1e-6, 1.0 - 1e-6)
    U = (-0.8090 * sto
         + 4.4875
         - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
         - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
         + 17.5842 * np.tanh(15.9308 * (sto - 0.3120)))
    return U


def OCP_LFP_Afshar2017(sto):
    """Open-circuit potential for LFP (LiFePO4) positive electrode.
    Afshar et al. 2017, used in PyBaMM Prada2013 parameter set.
    Flat plateau at ~3.4V.
    """
    sto = np.clip(sto, 1e-6, 1.0 - 1e-6)
    return 3.4077 - 0.020269 * sto + 0.5 * np.exp(-150.0 * sto) - 0.9 * np.exp(-30.0 * (1.0 - sto))


def OCP_LFP_Prada2013(sto):
    """Open-circuit potential for LFP (LiFePO4) positive electrode (Prada2013).
    U_ck = 3.4323
           - 0.8428 * exp(-80.2493*(1-y)^1.3198)
           - 3.2474e-6 * exp(20.2645*(1-y)^3.8003)
           + 3.2482e-6 * exp(20.2646*(1-y)^3.7995)
    """
    y = np.clip(sto, 1e-6, 1.0 - 1e-6)
    z = 1.0 - y
    U = (3.4323
         - 0.8428  * np.exp(-80.2493 * z**1.3198)
         - 3.2474e-6 * np.exp(20.2645 * z**3.8003)
         + 3.2482e-6 * np.exp(20.2646 * z**3.7995))
    return U


def OCP_LMFP(sto):
    """Open-circuit potential for LMFP (LiMn0.6Fe0.4PO4).
    Empirical two-plateau model: Fe²⁺/Fe³⁺ at ~3.45V, Mn²⁺/Mn³⁺ at ~4.1V.
    Composition: Mn₀.₆Fe₀.₄ → Fe plateau spans x≈0-0.4, Mn plateau x≈0.4-1.0.
    Based on published LMFP discharge profiles (Molenda 2017, Hu 2019).
    """
    sto = np.asarray(sto, dtype=np.float64)
    x = np.clip(sto, 1e-4, 1.0 - 1e-4)

    # Two-plateau construction using smooth transitions
    # High voltage region (Mn redox, 4.1V plateau, 60% of capacity)
    U_Mn = 4.10 - 0.02 * np.tanh(15.0 * (x - 0.05))  # initial drop-in
    # Transition between Mn and Fe plateaus around x=0.6
    transition = 0.5 * (1.0 + np.tanh(12.0 * (x - 0.60)))
    # Fe redox plateau at 3.45V
    U_Fe = 3.45 - 0.02 * np.tanh(15.0 * (x - 0.95))
    # Blend the two plateaus
    U = U_Mn * (1.0 - transition) + U_Fe * transition
    # End-of-discharge drop
    U = U - 0.3 * np.exp(-40.0 * (1.0 - x))
    # Start-of-discharge adjustment
    U = U + 0.2 * np.exp(-30.0 * x)
    return U


def OCP_LMFP64(sto):
    """OCP for LiMn0.6Fe0.4PO4 (LMFP64). Fe=40% capacity → transition at x≈0.40."""
    sto = np.asarray(sto, dtype=np.float64)
    x = np.clip(sto, 1e-4, 1.0 - 1e-4)
    U_Mn = 4.10 - 0.02 * np.tanh(15.0 * (x - 0.05))
    transition = 0.5 * (1.0 + np.tanh(12.0 * (x - 0.60)))
    U_Fe = 3.45 - 0.02 * np.tanh(15.0 * (x - 0.95))
    U = U_Mn * (1.0 - transition) + U_Fe * transition
    U = U - 0.3 * np.exp(-40.0 * (1.0 - x))
    U = U + 0.2 * np.exp(-30.0 * x)
    return U


def OCP_LMFP73(sto):
    """OCP for LiMn0.7Fe0.3PO4 (LMFP73). Fe=30% capacity → transition at x≈0.70."""
    sto = np.asarray(sto, dtype=np.float64)
    x = np.clip(sto, 1e-4, 1.0 - 1e-4)
    U_Mn = 4.10 - 0.02 * np.tanh(15.0 * (x - 0.05))
    transition = 0.5 * (1.0 + np.tanh(12.0 * (x - 0.70)))
    U_Fe = 3.45 - 0.02 * np.tanh(15.0 * (x - 0.95))
    U = U_Mn * (1.0 - transition) + U_Fe * transition
    U = U - 0.3 * np.exp(-40.0 * (1.0 - x))
    U = U + 0.2 * np.exp(-30.0 * x)
    return U


def OCP_LMFP82(sto):
    """OCP for LiMn0.8Fe0.2PO4 (LMFP82). Fe=20% capacity → transition at x≈0.80."""
    sto = np.asarray(sto, dtype=np.float64)
    x = np.clip(sto, 1e-4, 1.0 - 1e-4)
    U_Mn = 4.10 - 0.02 * np.tanh(15.0 * (x - 0.05))
    transition = 0.5 * (1.0 + np.tanh(12.0 * (x - 0.80)))
    U_Fe = 3.45 - 0.02 * np.tanh(15.0 * (x - 0.95))
    U = U_Mn * (1.0 - transition) + U_Fe * transition
    U = U - 0.3 * np.exp(-40.0 * (1.0 - x))
    U = U + 0.2 * np.exp(-30.0 * x)
    return U


# =============================================================================
# 5. ELECTROLYTE PROPERTY FUNCTIONS (Nyman2008)
# =============================================================================

def electrolyte_conductivity(c_e):
    """Electrolyte ionic conductivity [S/m] as function of c_e [mol/m^3].
    Nyman2008: kappa(c) with c in mol/L.
    """
    c_e = np.clip(c_e, 1.0, 5000.0)
    c = c_e / 1000.0  # mol/m^3 -> mol/L
    return 0.1297 * c**3 - 2.51 * c**1.5 + 3.329 * c


def electrolyte_diffusivity(c_e):
    """Electrolyte diffusivity [m^2/s] as function of c_e [mol/m^3].
    Nyman2008.
    """
    c_e = np.clip(c_e, 1.0, 5000.0)
    c = c_e / 1000.0
    return 8.794e-11 * c**2 - 3.972e-10 * c + 4.862e-10


# =============================================================================
# 6. PERCOLATION THEORY FOR COMPOSITE ELECTRODE CONDUCTIVITY
# =============================================================================

def composite_conductivity(wt_AM, wt_CA, wt_Binder, rho_AM,
                           rho_CA=2.0, rho_Binder=1.78,
                           sigma_CA=5000.0, phi_c=0.02, t_exp=1.5):
    """Percolation theory: electronic conductivity of composite electrode [S/m]."""
    vol_AM = wt_AM / rho_AM
    vol_CA = wt_CA / rho_CA
    vol_Binder = wt_Binder / rho_Binder
    vol_total = vol_AM + vol_CA + vol_Binder
    phi_CA = vol_CA / vol_total

    if phi_CA <= phi_c:
        return 1e-4
    return sigma_CA * ((phi_CA - phi_c) / (1.0 - phi_c)) ** t_exp


def recipe_to_electrode_props(wt_AM, wt_CA, wt_Binder, rho_AM,
                              porosity, R_p,
                              rho_CA=2.0, rho_Binder=1.78,
                              sigma_CA=5000.0, phi_c=0.02, t_exp=1.5):
    """Convert electrode recipe (weight %) to physical properties.

    Given the weight fractions of active material (AM), conductive additive (CA),
    and binder, plus the electrode porosity (set during calendaring), compute:
      - eps_s: active material volume fraction
      - eps_e: porosity (electrolyte volume fraction) — passed through
      - sigma_eff: effective electronic conductivity via percolation theory
      - a: specific interfacial area [1/m]

    Parameters
    ----------
    wt_AM, wt_CA, wt_Binder : float
        Weight percentages (should sum to ~100).
    rho_AM : float
        Active material density [g/cm³].
    porosity : float
        Electrode porosity (eps_e), typically 0.25-0.45.
    R_p : float
        Particle radius [m].
    rho_CA, rho_Binder : float
        Densities of CA and binder [g/cm³].

    Returns
    -------
    dict with keys: eps_s, eps_e, sigma_eff, a, phi_CA, phi_AM
    """
    # Volume fractions within the solid phase
    vol_AM = wt_AM / rho_AM
    vol_CA = wt_CA / rho_CA
    vol_Binder = wt_Binder / rho_Binder
    vol_total = vol_AM + vol_CA + vol_Binder

    phi_AM = vol_AM / vol_total       # AM volume fraction in solid
    phi_CA = vol_CA / vol_total       # CA volume fraction in solid
    phi_Binder = vol_Binder / vol_total

    # Actual volume fractions in electrode (solid_fraction = 1 - porosity)
    solid_fraction = 1.0 - porosity
    eps_s = phi_AM * solid_fraction   # active material volume fraction
    eps_CA = phi_CA * solid_fraction
    eps_e = porosity

    # Specific interfacial area: a = 3 * eps_s / R_p
    a = 3.0 * eps_s / R_p

    # Electronic conductivity via percolation theory (CA network)
    if phi_CA <= phi_c:
        sigma_eff = 1e-4  # below percolation threshold
    else:
        sigma_eff = sigma_CA * ((phi_CA - phi_c) / (1.0 - phi_c)) ** t_exp

    return {
        "eps_s": eps_s,
        "eps_e": eps_e,
        "sigma_eff": sigma_eff,
        "a": a,
        "phi_AM": phi_AM,
        "phi_CA": phi_CA,
        "phi_Binder": phi_Binder,
        "eps_CA": eps_CA,
    }


# =============================================================================
# 6b. CA MATERIAL DATABASE + BAUHOFER-KOVACS BINARY PERCOLATION MODEL
# =============================================================================

CA_PARAMS = {
    "SuperP": {
        "name": "Super P (Carbon Black, 0D)",
        "sigma": 5000.0,    # S/m  — Spahr et al. 2011
        "rho":   2.0,       # g/cm³
        "phi_c": 0.02,      # percolation threshold (vol fraction in solid)
        "t":     1.5,       # transport exponent
        "geometry": "0D",
    },
    "VGCF": {
        "name": "VGCF (기상성장 탄소섬유, 1D)",
        "sigma": 30000.0,   # S/m  — Endo et al. 2008
        "rho":   1.9,
        "phi_c": 0.005,     # low threshold due to high aspect ratio (~100)
        "t":     1.5,
        "geometry": "1D",
    },
    "MWCNT": {
        "name": "MWCNT (다중벽 탄소나노튜브, 1D)",
        "sigma": 100000.0,  # S/m  — intrinsic, composite values lower
        "rho":   1.5,
        "phi_c": 0.001,     # very low threshold, aspect ratio ~500
        "t":     1.3,       # lower exponent for fibrous network
        "geometry": "1D",
    },
    "Graphene": {
        "name": "Graphene / rGO (2D)",
        "sigma": 50000.0,
        "rho":   2.2,
        "phi_c": 0.005,
        "t":     1.4,
        "geometry": "2D",
    },
    "KS6": {
        "name": "KS6 (플레이크 흑연, 2D)",
        "sigma": 2000.0,
        "rho":   2.2,
        "phi_c": 0.05,      # high threshold due to rigid platelets
        "t":     2.0,
        "geometry": "2D",
    },
}


def recipe_to_electrode_props_bk(
    wt_AM, wt_CA1, wt_CA2, wt_Binder, rho_AM,
    porosity, R_p,
    ca1_type="SuperP", ca2_type=None,
    rho_Binder=1.78,
):
    """Bauhofer-Kovacs (2009) binary filler percolation model.

    Single CA (ca2_type=None or wt_CA2=0):
        σ_eff = σ_CA · ((φ - φ_c) / (1 - φ_c))^t          for φ > φ_c

    Binary CA (Bauhofer & Kovacs, Compos. Sci. Technol. 69, 1486, 2009):
        Percolation condition : φ₁/φ_c1 + φ₂/φ_c2 ≥ 1
        Effective thresholds  : φ_c1_eff = φ_c1 · max(0, 1 − φ₂/φ_c2)
                                φ_c2_eff = φ_c2 · max(0, 1 − φ₁/φ_c1)
        Excess fractions      : ψᵢ = max(0, φᵢ − φ_ci_eff)
        Conductivity          : σ_eff = Σ σᵢ · (ψᵢ/(1 − φ_ci))^tᵢ

    The B-K model captures the synergistic threshold reduction that occurs when
    0D (CB) fills the gaps in a 1D (CNT/VGCF) percolating network.
    """
    p1 = CA_PARAMS[ca1_type]

    vol_AM     = wt_AM     / rho_AM
    vol_CA1    = wt_CA1    / p1["rho"]
    vol_Binder = wt_Binder / rho_Binder

    use_binary = (ca2_type is not None) and (wt_CA2 > 0)
    if use_binary:
        p2 = CA_PARAMS[ca2_type]
        vol_CA2 = wt_CA2 / p2["rho"]
    else:
        p2 = None
        vol_CA2 = 0.0

    vol_total = vol_AM + vol_CA1 + vol_CA2 + vol_Binder

    phi_AM  = vol_AM  / vol_total
    phi_CA1 = vol_CA1 / vol_total
    phi_CA2 = vol_CA2 / vol_total

    solid_fraction = 1.0 - porosity
    eps_s = phi_AM * solid_fraction
    eps_e = porosity
    a     = 3.0 * eps_s / R_p

    sigma1, phi_c1, t1 = p1["sigma"], p1["phi_c"], p1["t"]

    if not use_binary:
        # ── Single CA ─────────────────────────────────────────────────────────
        perc_index = phi_CA1 / phi_c1
        if phi_CA1 <= phi_c1:
            sigma_eff = 1e-4
        else:
            sigma_eff = sigma1 * ((phi_CA1 - phi_c1) / (1.0 - phi_c1)) ** t1
        phi_c1_eff = phi_c1
        phi_c2_eff = None
    else:
        # ── Binary CA: Bauhofer-Kovacs ────────────────────────────────────────
        sigma2, phi_c2, t2 = p2["sigma"], p2["phi_c"], p2["t"]

        perc_index = phi_CA1 / phi_c1 + phi_CA2 / phi_c2

        if perc_index <= 1.0:
            sigma_eff  = 1e-4
            phi_c1_eff = phi_c1
            phi_c2_eff = phi_c2
        else:
            # Effective thresholds: each reduced by the other filler's contribution
            phi_c1_eff = phi_c1 * max(0.0, 1.0 - phi_CA2 / phi_c2)
            phi_c2_eff = phi_c2 * max(0.0, 1.0 - phi_CA1 / phi_c1)

            psi1 = max(0.0, phi_CA1 - phi_c1_eff)
            psi2 = max(0.0, phi_CA2 - phi_c2_eff)

            s1 = sigma1 * (psi1 / (1.0 - phi_c1)) ** t1 if psi1 > 0 else 0.0
            s2 = sigma2 * (psi2 / (1.0 - phi_c2)) ** t2 if psi2 > 0 else 0.0
            sigma_eff = s1 + s2

    return {
        "eps_s":       eps_s,
        "eps_e":       eps_e,
        "sigma_eff":   sigma_eff,
        "a":           a,
        "phi_AM":      phi_AM,
        "phi_CA1":     phi_CA1,
        "phi_CA2":     phi_CA2,
        "perc_index":  perc_index,
        "phi_c1_eff":  phi_c1_eff,
        "phi_c2_eff":  phi_c2_eff,
    }


# =============================================================================
# 7. MESH ASSEMBLY
# =============================================================================

def build_cell_mesh(p, N_n=20, N_s=10, N_p=20, N_r=10):
    """Build the complete cell mesh for x-direction and r-direction particles."""

    # x-direction meshes for each domain
    mesh_n = build_uniform_mesh(p["L_n"], N_n)
    mesh_s = build_uniform_mesh(p["L_s"], N_s)
    mesh_p = build_uniform_mesh(p["L_p"], N_p)

    N_tot = N_n + N_s + N_p

    # Concatenated x-nodes (shifted to global coordinates)
    x_nodes_n = mesh_n["nodes"]
    x_nodes_s = mesh_s["nodes"] + p["L_n"]
    x_nodes_p = mesh_p["nodes"] + p["L_n"] + p["L_s"]
    x_nodes = np.concatenate([x_nodes_n, x_nodes_s, x_nodes_p])

    # x-direction edges for the full domain
    x_edges_n = mesh_n["edges"]
    x_edges_s = mesh_s["edges"] + p["L_n"]
    x_edges_p = mesh_p["edges"] + p["L_n"] + p["L_s"]

    # Concatenated edges (remove duplicates at interfaces)
    x_edges = np.concatenate([x_edges_n, x_edges_s[1:], x_edges_p[1:]])
    x_d_edges = np.diff(x_edges)  # Should be N_tot values... but edges are N_tot+1
    # Actually: x_edges has N_n+1 + N_s + N_p = N_tot+1 edges
    # x_d_edges has N_tot values = cell widths

    # Node-to-node distances for full domain
    x_d_nodes = np.diff(x_nodes)  # (N_tot - 1,)

    # Build global gradient and divergence matrices for x-direction
    inv_d_nodes = 1.0 / x_d_nodes
    G_x = sparse.diags([-inv_d_nodes, inv_d_nodes], [0, 1],
                        shape=(N_tot - 1, N_tot), format="csr")

    inv_d_edges = 1.0 / x_d_edges
    D_x = sparse.diags([-inv_d_edges, inv_d_edges], [0, 1],
                        shape=(N_tot, N_tot + 1), format="csr")

    # Sub-domain gradient/divergence for neg and pos electrodes
    G_n = build_gradient_matrix(mesh_n)
    G_p = build_gradient_matrix(mesh_p)
    D_n = build_divergence_matrix_cartesian(mesh_n)
    D_p = build_divergence_matrix_cartesian(mesh_p)

    # Particle (r-direction) FVM
    particle_n = build_spherical_fvm(p["R_p_n"], N_r)
    particle_p = build_spherical_fvm(p["R_p_p"], N_r)

    # Porosity array (full x-domain)
    eps_e = np.concatenate([
        np.full(N_n, p["eps_e_n"]),
        np.full(N_s, p["eps_e_s"]),
        np.full(N_p, p["eps_e_p"])
    ])

    # Bruggeman-corrected porosity for electrolyte transport
    eps_e_brugg = np.concatenate([
        np.full(N_n, p["eps_e_n"] ** p["brugg_n"]),
        np.full(N_s, p["eps_e_s"] ** p["brugg_s"]),
        np.full(N_p, p["eps_e_p"] ** p["brugg_p"])
    ])

    return {
        "N_n": N_n, "N_s": N_s, "N_p": N_p, "N_tot": N_tot, "N_r": N_r,
        "mesh_n": mesh_n, "mesh_s": mesh_s, "mesh_p": mesh_p,
        "x_nodes": x_nodes, "x_edges": x_edges,
        "x_d_edges": x_d_edges, "x_d_nodes": x_d_nodes,
        "G_x": G_x, "D_x": D_x,
        "G_n": G_n, "G_p": G_p, "D_n": D_n, "D_p": D_p,
        "particle_n": particle_n, "particle_p": particle_p,
        "eps_e": eps_e, "eps_e_brugg": eps_e_brugg,
    }


# =============================================================================
# 8. STATE VECTOR PACKING / UNPACKING
# =============================================================================

def build_slices(geo):
    """Define index slices for each variable in the state vector y."""
    N_n, N_p, N_s, N_tot, N_r = geo["N_n"], geo["N_p"], geo["N_s"], geo["N_tot"], geo["N_r"]

    idx = 0
    sl = {}

    sl["c_s_n"] = slice(idx, idx + N_n * N_r);  idx += N_n * N_r
    sl["c_s_p"] = slice(idx, idx + N_p * N_r);  idx += N_p * N_r
    sl["eps_c_e"] = slice(idx, idx + N_tot);     idx += N_tot
    sl["phi_s_n"] = slice(idx, idx + N_n);       idx += N_n
    sl["phi_s_p"] = slice(idx, idx + N_p);       idx += N_p
    sl["phi_e"] = slice(idx, idx + N_tot);       idx += N_tot

    sl["total"] = idx
    return sl


def unpack(y, sl, geo):
    """Unpack state vector into named arrays."""
    N_n, N_p, N_r = geo["N_n"], geo["N_p"], geo["N_r"]

    c_s_n = y[sl["c_s_n"]].reshape(N_n, N_r)   # (N_n, N_r)
    c_s_p = y[sl["c_s_p"]].reshape(N_p, N_r)   # (N_p, N_r)
    eps_c_e = y[sl["eps_c_e"]]                   # (N_tot,)
    phi_s_n = y[sl["phi_s_n"]]                   # (N_n,)
    phi_s_p = y[sl["phi_s_p"]]                   # (N_p,)
    phi_e = y[sl["phi_e"]]                       # (N_tot,)

    return c_s_n, c_s_p, eps_c_e, phi_s_n, phi_s_p, phi_e


# =============================================================================
# 9. SPHERICAL PARTICLE DIFFUSION RHS
# =============================================================================

def particle_rhs(c_s_2d, j_array, D_s, particle_mesh, a_s, c_s_max):
    """Compute dc_s/dt for all particles in one electrode.

    c_s_2d: (N_x, N_r) concentration inside particles
    j_array: (N_x,) interfacial current density [A/m^2] at each x-node
    D_s: solid diffusivity [m^2/s], scalar OR callable(c_s, c_s_max)->scalar
    particle_mesh: spherical FVM mesh dict
    a_s: specific interfacial area [1/m]

    Returns: dc_s_dt (N_x, N_r)
    """
    N_x, N_r = c_s_2d.shape
    G = particle_mesh["G"]        # (N_r-1, N_r) gradient matrix
    D = particle_mesh["D"]        # (N_r, N_r+1) divergence matrix
    r2_edges = particle_mesh["r2_edges"]  # (N_r+1,)
    R_p = particle_mesh["R_p"]

    dc_dt = np.zeros_like(c_s_2d)

    for ix in range(N_x):
        c_s = c_s_2d[ix, :]  # (N_r,)

        # D_s may be concentration-dependent (callable) or constant (scalar)
        if callable(D_s):
            D_s_val = D_s(c_s, c_s_max)
        else:
            D_s_val = D_s

        # Internal fluxes: N = -D_s * grad(c_s)  at internal edges
        grad_cs = G @ c_s          # (N_r-1,)
        flux_internal = -D_s_val * grad_cs  # (N_r-1,)

        # Boundary fluxes
        # Left (r=0): symmetry -> N = 0
        flux_left = 0.0

        # Right (r=R_p): -D_s * dc_s/dr = j/F
        # PyBaMM convention (∇·(σ∇φ_s) = a·j):
        #   j > 0 at anode during discharge (Li leaving solid, anodic)
        #   j < 0 at cathode during discharge (Li entering solid, cathodic)
        # Molar flux outward = j/F
        flux_right = j_array[ix] / F

        # Full edge fluxes (N_r+1,) — NOTE: these are radial fluxes, not yet r^2 weighted
        flux_edges = np.concatenate(([flux_left], flux_internal, [flux_right]))

        # Divergence: D already includes r^2/volume weighting
        # D @ (r^2 * flux) would double-count because D already has r^2 built in
        # Actually, PyBaMM: divergence_matrix @ (r_edges^2 * flux)
        # But our D matrix already includes r^2 in its construction.
        # Let me reconsider...
        #
        # PyBaMM FVM for spherical:
        #   div(N) at node i = [r_{i+1}^2 * N_{i+1} - r_i^2 * N_i] / V_i
        #   where V_i = (r_{i+1}^3 - r_i^3) / 3
        #
        # Our D matrix: D[i,i] = -r_left^2/V, D[i,i+1] = +r_right^2/V
        # So D @ flux_edges gives exactly the spherical divergence.

        div_flux = D @ flux_edges   # (N_r,)

        # dc_s/dt = -div(N_s)  (note: N_s is already -D*grad, so div is positive for outflow)
        dc_dt[ix, :] = -div_flux

    return dc_dt


# =============================================================================
# 10. BUTLER-VOLMER KINETICS
# =============================================================================

def butler_volmer(phi_s, phi_e, U_eq, c_e, c_s_surf, c_s_max, k_rate, T):
    """Symmetric Butler-Volmer: j = 2*j0*sinh(F*eta/(2RT)).
    Returns j [A/m^2] (interfacial current density, NOT volumetric).
    Includes smooth reduction for low c_e to prevent singularity.
    """
    # Exchange current density
    c_e_safe = np.clip(c_e, 1.0, None)
    c_s_safe = np.clip(c_s_surf, 1.0, c_s_max - 1.0)

    # m_ref (= k_rate in Chen2020) already in (A/m2)(m3/mol)^1.5 — no F needed
    j0 = k_rate * np.sqrt(c_e_safe) * np.sqrt(c_s_safe) * np.sqrt(c_s_max - c_s_safe)

    # Smooth reduction when c_e is low (prevents numerical blowup)
    # Smoothly ramps from 0 at c_e=0 to 1 at c_e=100
    ce_factor = np.where(c_e < 100.0, np.clip(c_e / 100.0, 0.0, 1.0), 1.0)
    j0 = j0 * ce_factor

    # Overpotential
    eta = phi_s - phi_e - U_eq
    eta = np.clip(eta, -2.0, 2.0)  # Prevent overflow

    # Symmetric BV
    j = 2.0 * j0 * np.sinh(0.5 * F * eta / (R_gas * T))

    return j


# =============================================================================
# 11. MAIN RHS FUNCTION
# =============================================================================

def solve_consistent_ic(p, geo, sl, y0, I_app):
    """Solve for consistent initial conditions of algebraic variables (phi_s, phi_e).

    At t=0, the ODE variables (c_s, c_e) are fixed. We find phi_s, phi_e such that
    the algebraic residuals are zero. This mimics PyBaMM's IDA initialization.
    """
    from scipy.optimize import least_squares

    N_n = geo["N_n"]
    N_s = geo["N_s"]
    N_p = geo["N_p"]
    N_tot = geo["N_tot"]
    N_r = geo["N_r"]
    eps_e = geo["eps_e"]

    i_app = I_app / p["A_cell"]
    chi_factor = 2.0 * (1.0 - p["t_plus"]) * p["therm_factor"]

    # Fixed ODE variables
    c_s_n = y0[sl["c_s_n"]].reshape(N_n, N_r)
    c_s_p = y0[sl["c_s_p"]].reshape(N_p, N_r)
    eps_c_e = y0[sl["eps_c_e"]]
    c_e = eps_c_e / eps_e
    c_e = np.clip(c_e, 1.0, 5000.0)

    c_s_surf_n = c_s_n[:, -1]
    c_s_surf_p = c_s_p[:, -1]

    # Properties
    kappa = p["kappa_e_func"](c_e)
    kappa_eff = kappa * geo["eps_e_brugg"]

    sto_n = c_s_surf_n / p["c_s_max_n"]
    sto_p = c_s_surf_p / p["c_s_max_p"]
    U_n = p["OCP_n"](sto_n)
    U_p = p["OCP_p"](sto_p)

    # Initial guess for algebraic variables
    x0 = np.concatenate([y0[sl["phi_s_n"]], y0[sl["phi_s_p"]], y0[sl["phi_e"]]])

    G_n = geo["G_n"]
    G_p = geo["G_p"]
    D_n = geo["D_n"]
    D_p = geo["D_p"]
    G_x = geo["G_x"]
    D_x = geo["D_x"]

    sigma_eff_n = p["sigma_eff_n"]
    sigma_eff_p = p["sigma_eff_p"]

    # Scale factors for residuals (PyBaMM uses L_x^2)
    L_x = p["L_n"] + p["L_s"] + p["L_p"]
    scale = L_x**2

    def residual(x):
        phi_s_n = x[:N_n]
        phi_s_p = x[N_n:N_n + N_p]
        phi_e = x[N_n + N_p:]

        phi_e_n = phi_e[:N_n]
        phi_e_p = phi_e[N_n + N_s:]

        j_n = butler_volmer(phi_s_n, phi_e_n, U_n, c_e[:N_n],
                            c_s_surf_n, p["c_s_max_n"], p["k_n"], p["T"])
        j_p = butler_volmer(phi_s_p, phi_e_p, U_p, c_e[N_n + N_s:],
                            c_s_surf_p, p["c_s_max_p"], p["k_p"], p["T"])

        aj_n = p["a_n"] * j_n
        aj_p = p["a_p"] * j_p
        a_j_full = np.zeros(N_tot)
        a_j_full[:N_n] = aj_n
        a_j_full[N_n + N_s:] = aj_p

        # Solid potential residuals: ∇·(σ∇φ_s) = a·j (PyBaMM convention)
        grad_ps_n = G_n @ phi_s_n
        flux_n = apply_neumann_bc(sigma_eff_n * grad_ps_n, 0.0, 0.0)
        res_n = scale * (D_n @ flux_n - aj_n)
        res_n[0] = phi_s_n[0] - 0.0  # Dirichlet: φ_s_n[0] = 0

        grad_ps_p = G_p @ phi_s_p
        flux_p = apply_neumann_bc(sigma_eff_p * grad_ps_p, 0.0, -i_app)
        res_p = scale * (D_p @ flux_p - aj_p)

        # Electrolyte potential residuals: ∇·(i_e) = a·j → div(i_e) - a·j = 0
        grad_pe = G_x @ phi_e
        grad_ce = G_x @ c_e
        kappa_eff_edges = 0.5 * (kappa_eff[:-1] + kappa_eff[1:])
        c_e_edges = np.clip(0.5 * (c_e[:-1] + c_e[1:]), 1.0, None)
        chi_RT_Fc = chi_factor * R_gas * p["T"] / (F * c_e_edges)
        i_e_int = kappa_eff_edges * (chi_RT_Fc * grad_ce - grad_pe)
        i_e_edges = apply_neumann_bc(i_e_int, 0.0, 0.0)
        res_e = scale * (D_x @ i_e_edges - a_j_full)
        # No pin on phi_e — BV coupling to phi_s uniquely determines phi_e

        return np.concatenate([res_n, res_p, res_e])

    sol = least_squares(residual, x0, method='lm', ftol=1e-10, xtol=1e-10)
    print(f"  IC solver: cost={sol.cost:.2e}, nfev={sol.nfev}")

    y0[sl["phi_s_n"]] = sol.x[:N_n]
    y0[sl["phi_s_p"]] = sol.x[N_n:N_n + N_p]
    y0[sl["phi_e"]] = sol.x[N_n + N_p:]
    return y0


def solve_potentials(phi_guess, c_e, c_s_surf_n, c_s_surf_p, p, geo, I_app):
    """Solve the algebraic potential equations at a given time instant.

    Given concentrations (c_e, c_s_surf), find phi_s_n, phi_s_p, phi_e
    such that charge conservation holds. Returns potentials and j values.
    """
    from scipy.optimize import least_squares

    N_n = geo["N_n"]
    N_s = geo["N_s"]
    N_p = geo["N_p"]
    N_tot = geo["N_tot"]

    G_n, G_p = geo["G_n"], geo["G_p"]
    D_n, D_p = geo["D_n"], geo["D_p"]
    G_x, D_x = geo["G_x"], geo["D_x"]

    sigma_eff_n = p["sigma_eff_n"]
    sigma_eff_p = p["sigma_eff_p"]
    i_app = I_app / p["A_cell"]
    chi_factor = 2.0 * (1.0 - p["t_plus"]) * p["therm_factor"]

    kappa = p["kappa_e_func"](c_e)
    kappa_eff = kappa * geo["eps_e_brugg"]

    sto_n = c_s_surf_n / p["c_s_max_n"]
    sto_p = c_s_surf_p / p["c_s_max_p"]
    U_n = p["OCP_n"](sto_n)
    U_p = p["OCP_p"](sto_p)

    L_x = p["L_n"] + p["L_s"] + p["L_p"]
    scale = L_x**2

    def residual(x):
        phi_s_n = x[:N_n]
        phi_s_p = x[N_n:N_n + N_p]
        phi_e = x[N_n + N_p:]

        j_n = butler_volmer(phi_s_n, phi_e[:N_n], U_n, c_e[:N_n],
                            c_s_surf_n, p["c_s_max_n"], p["k_n"], p["T"])
        j_p = butler_volmer(phi_s_p, phi_e[N_n + N_s:], U_p, c_e[N_n + N_s:],
                            c_s_surf_p, p["c_s_max_p"], p["k_p"], p["T"])

        aj_n = p["a_n"] * j_n
        aj_p = p["a_p"] * j_p
        a_j_full = np.zeros(N_tot)
        a_j_full[:N_n] = aj_n
        a_j_full[N_n + N_s:] = aj_p

        # phi_s_n
        flux_n = apply_neumann_bc(sigma_eff_n * (G_n @ phi_s_n), i_app, 0.0)
        res_n = scale * (D_n @ flux_n + aj_n)
        res_n[0] = phi_s_n[0]  # Dirichlet: phi_s_n[0] = 0

        # phi_s_p
        flux_p = apply_neumann_bc(sigma_eff_p * (G_p @ phi_s_p), 0.0, -i_app)
        res_p = scale * (D_p @ flux_p + aj_p)

        # phi_e
        grad_ce = G_x @ c_e
        grad_pe = G_x @ phi_e
        kap_edges = 0.5 * (kappa_eff[:-1] + kappa_eff[1:])
        ce_edges = np.clip(0.5 * (c_e[:-1] + c_e[1:]), 1.0, None)
        chi_term = chi_factor * R_gas * p["T"] / (F * ce_edges)
        i_e = kap_edges * (chi_term * grad_ce - grad_pe)
        i_e_edges = apply_neumann_bc(i_e, 0.0, 0.0)
        res_e = scale * (D_x @ i_e_edges + a_j_full)
        res_e[0] = phi_e[0] - (-U_n[0])

        return np.concatenate([res_n, res_p, res_e])

    sol = least_squares(residual, phi_guess, method='lm', ftol=1e-10, xtol=1e-10,
                        max_nfev=200)

    phi_s_n = sol.x[:N_n]
    phi_s_p = sol.x[N_n:N_n + N_p]
    phi_e = sol.x[N_n + N_p:]

    j_n = butler_volmer(phi_s_n, phi_e[:N_n], U_n, c_e[:N_n],
                        c_s_surf_n, p["c_s_max_n"], p["k_n"], p["T"])
    j_p = butler_volmer(phi_s_p, phi_e[N_n + N_s:], U_p, c_e[N_n + N_s:],
                        c_s_surf_p, p["c_s_max_p"], p["k_p"], p["T"])

    return phi_s_n, phi_s_p, phi_e, j_n, j_p


def build_rhs(p, geo, sl, I_app, tau=1e-3):
    """Build the RHS function f(t, y) for solve_ivp.

    Strategy: Pseudo-transient DAE.
    ODE variables (c_s, c_e): standard RHS with physical time derivatives.
    Algebraic variables (phi_s, phi_e): residual / tau (fast relaxation).
    BDF handles the stiffness of the pseudo-transient implicitly.
    """
    N_n = geo["N_n"]
    N_s = geo["N_s"]
    N_p = geo["N_p"]
    N_tot = geo["N_tot"]
    N_r = geo["N_r"]

    G_x, D_x = geo["G_x"], geo["D_x"]
    G_n, D_n = geo["G_n"], geo["D_n"]
    G_p, D_p = geo["G_p"], geo["D_p"]
    eps_e = geo["eps_e"]

    OCP_n_func = p["OCP_n"]
    OCP_p_func = p["OCP_p"]
    eps_e_brugg = geo["eps_e_brugg"]

    particle_n = geo["particle_n"]
    particle_p = geo["particle_p"]

    sigma_eff_n = p["sigma_eff_n"]
    sigma_eff_p = p["sigma_eff_p"]
    i_app = I_app / p["A_cell"]
    chi_factor = 2.0 * (1.0 - p["t_plus"]) * p["therm_factor"]
    L_x = p["L_n"] + p["L_s"] + p["L_p"]
    scale = L_x**2
    kappa_e_func = p["kappa_e_func"]
    D_e_func = p["D_e_func"]

    inv_tau = 1.0 / tau

    def rhs(t, y):
        # ---- Unpack full state vector ----
        c_s_n, c_s_p, eps_c_e, phi_s_n, phi_s_p, phi_e = unpack(y, sl, geo)

        c_e = eps_c_e / eps_e
        c_e = np.clip(c_e, 1.0, 5000.0)

        c_s_surf_n = np.clip(c_s_n[:, -1], 1.0, p["c_s_max_n"] - 1.0)
        c_s_surf_p = np.clip(c_s_p[:, -1], 1.0, p["c_s_max_p"] - 1.0)

        # ---- OCP ----
        sto_n = c_s_surf_n / p["c_s_max_n"]
        sto_p = c_s_surf_p / p["c_s_max_p"]
        U_n = OCP_n_func(sto_n)
        U_p = OCP_p_func(sto_p)

        # ---- Butler-Volmer kinetics ----
        phi_e_n = phi_e[:N_n]
        phi_e_p = phi_e[N_n + N_s:]
        j_n = butler_volmer(phi_s_n, phi_e_n, U_n, c_e[:N_n],
                            c_s_surf_n, p["c_s_max_n"], p["k_n"], p["T"])
        j_p = butler_volmer(phi_s_p, phi_e_p, U_p, c_e[N_n + N_s:],
                            c_s_surf_p, p["c_s_max_p"], p["k_p"], p["T"])

        # Volumetric source
        aj_n = p["a_n"] * j_n
        aj_p = p["a_p"] * j_p
        a_j_full = np.zeros(N_tot)
        a_j_full[:N_n] = aj_n
        a_j_full[N_n + N_s:] = aj_p

        # ================================================================
        # ODE 1: Solid particle diffusion (spherical FVM)
        # ================================================================
        dc_s_n_dt = particle_rhs(c_s_n, j_n, p["D_s_n"], particle_n,
                                 p["a_n"], p["c_s_max_n"])
        dc_s_p_dt = particle_rhs(c_s_p, j_p, p["D_s_p"], particle_p,
                                 p["a_p"], p["c_s_max_p"])

        # ================================================================
        # ODE 2: Electrolyte concentration
        # ================================================================
        D_e_val = D_e_func(c_e)
        D_e_eff = D_e_val * eps_e_brugg
        grad_ce = G_x @ c_e
        D_eff_edges = 0.5 * (D_e_eff[:-1] + D_e_eff[1:])
        flux_ce = apply_neumann_bc(D_eff_edges * grad_ce, 0.0, 0.0)
        div_flux_ce = D_x @ flux_ce
        source_ce = (1.0 - p["t_plus"]) / F * a_j_full
        d_eps_ce_dt = div_flux_ce + source_ce

        # ================================================================
        # Pseudo-transient: phi_s and phi_e residuals / tau
        # PyBaMM convention: ∇·(σ∇φ_s) = a·j, ∇·(i_e) = -a·j
        # ================================================================

        # phi_s_n: ∇·(σ_eff·∇φ_s) - a·j = 0
        grad_ps_n = G_n @ phi_s_n
        flux_ps_n = apply_neumann_bc(sigma_eff_n * grad_ps_n, 0.0, 0.0)
        res_phi_s_n = scale * (D_n @ flux_ps_n - aj_n)
        res_phi_s_n[0] = (0.0 - phi_s_n[0])  # Dirichlet: phi_s_n[0] = 0

        # phi_s_p: ∇·(σ_eff·∇φ_s) - a·j = 0
        grad_ps_p = G_p @ phi_s_p
        flux_ps_p = apply_neumann_bc(sigma_eff_p * grad_ps_p, 0.0, -i_app)
        res_phi_s_p = scale * (D_p @ flux_ps_p - aj_p)

        # phi_e: correct equation is ∇·(i_e) = a·j, i.e., div(i_e) - a·j = 0
        # For pseudo-transient stability, NEGATE the residual:
        #   dφ_e/dt = -(div(i_e) - a·j)/tau = (-div(i_e) + a·j)/tau
        # This is stable AND drives to the same steady state div(i_e) = a·j
        grad_pe = G_x @ phi_e
        kappa = kappa_e_func(c_e)
        kappa_eff = kappa * eps_e_brugg
        kap_edges = 0.5 * (kappa_eff[:-1] + kappa_eff[1:])
        ce_edges = np.clip(0.5 * (c_e[:-1] + c_e[1:]), 1.0, None)
        chi_term = chi_factor * R_gas * p["T"] / (F * ce_edges)
        i_e = kap_edges * (chi_term * grad_ce - grad_pe)
        i_e_edges = apply_neumann_bc(i_e, 0.0, 0.0)
        res_phi_e = scale * (-D_x @ i_e_edges + a_j_full)

        # ================================================================
        # Assemble full dydt
        # ================================================================
        dydt = np.empty_like(y)
        dydt[sl["c_s_n"]] = dc_s_n_dt.ravel()
        dydt[sl["c_s_p"]] = dc_s_p_dt.ravel()
        dydt[sl["eps_c_e"]] = d_eps_ce_dt
        dydt[sl["phi_s_n"]] = res_phi_s_n * inv_tau
        dydt[sl["phi_s_p"]] = res_phi_s_p * inv_tau
        dydt[sl["phi_e"]] = res_phi_e * inv_tau

        return dydt

    return rhs


# =============================================================================
# 12. INITIAL CONDITIONS
# =============================================================================

def build_initial_conditions(p, geo, sl):
    """Construct physically consistent initial state vector."""
    N_n, N_p, N_s, N_tot, N_r = geo["N_n"], geo["N_p"], geo["N_s"], geo["N_tot"], geo["N_r"]

    y0 = np.zeros(sl["total"])

    # Solid concentrations: uniform at initial values
    c_s_n_init = np.full((N_n, N_r), p["c_s_init_n"])
    c_s_p_init = np.full((N_p, N_r), p["c_s_init_p"])
    y0[sl["c_s_n"]] = c_s_n_init.ravel()
    y0[sl["c_s_p"]] = c_s_p_init.ravel()

    # Electrolyte: eps * c_e
    eps_e = geo["eps_e"]
    y0[sl["eps_c_e"]] = eps_e * p["c_e_init"]

    # Potentials: initialize from OCP (physically consistent)
    sto_n_init = p["c_s_init_n"] / p["c_s_max_n"]
    sto_p_init = p["c_s_init_p"] / p["c_s_max_p"]
    U_n_init = p["OCP_n"](sto_n_init)
    U_p_init = p["OCP_p"](sto_p_init)

    # phi_s_n = 0 (reference)
    y0[sl["phi_s_n"]] = 0.0

    # phi_e = -U_n (so that eta_n = phi_s_n - phi_e - U_n = 0 at equilibrium)
    y0[sl["phi_e"]] = -U_n_init

    # phi_s_p = U_p - U_n (so that V_cell = phi_s_p - phi_s_n ≈ U_p - U_n)
    y0[sl["phi_s_p"]] = U_p_init - U_n_init

    V_init = U_p_init - U_n_init
    print(f"Initial OCV: {V_init:.4f} V")
    print(f"  Anode  OCP (sto={sto_n_init:.4f}): {U_n_init:.4f} V")
    print(f"  Cathode OCP (sto={sto_p_init:.4f}): {U_p_init:.4f} V")

    return y0


# =============================================================================
# 13. VOLTAGE EXTRACTION
# =============================================================================

def cell_voltage(y, sl, geo, p):
    """Extract cell voltage V = phi_s_p(x=L) - phi_s_n(x=0)."""
    phi_s_n = y[sl["phi_s_n"]]
    phi_s_p = y[sl["phi_s_p"]]
    return phi_s_p[-1] - phi_s_n[0]


# =============================================================================
# 14. SPARSITY PATTERN FOR JACOBIAN
# =============================================================================

def build_jac_sparsity(sl, geo):
    """Build Jacobian sparsity pattern for solve_ivp.

    Since algebraic variables (phi_s, phi_e) are solved internally,
    their dy/dt = 0 always. The Jacobian only has nonzero entries for
    ODE variables (c_s_n, c_s_p, eps_c_e) rows/cols.
    But ODE RHS depends on ALL ODE variables through the internal algebraic solve.
    """
    n = sl["total"]
    N_n, N_p, N_s, N_tot, N_r = geo["N_n"], geo["N_p"], geo["N_s"], geo["N_tot"], geo["N_r"]

    rows, cols = [], []

    # c_s_n: tridiagonal in r for each x-node, plus coupling to c_e and other c_s (through j)
    for ix in range(N_n):
        r_start = sl["c_s_n"].start + ix * N_r
        for ir in range(N_r):
            row = r_start + ir
            # r-direction tridiagonal
            for dr in [-1, 0, 1]:
                if 0 <= ir + dr < N_r:
                    rows.append(row); cols.append(r_start + ir + dr)
            # Surface node depends on c_e (through j via internal solve)
            if ir == N_r - 1:
                rows.append(row); cols.append(sl["eps_c_e"].start + ix)
                # Also depends on other particles' surface concentration
                for jx in range(N_n):
                    rows.append(row); cols.append(sl["c_s_n"].start + jx * N_r + N_r - 1)

    # c_s_p: same structure
    for ix in range(N_p):
        r_start = sl["c_s_p"].start + ix * N_r
        for ir in range(N_r):
            row = r_start + ir
            for dr in [-1, 0, 1]:
                if 0 <= ir + dr < N_r:
                    rows.append(row); cols.append(r_start + ir + dr)
            if ir == N_r - 1:
                rows.append(row); cols.append(sl["eps_c_e"].start + N_n + N_s + ix)
                for jx in range(N_p):
                    rows.append(row); cols.append(sl["c_s_p"].start + jx * N_r + N_r - 1)

    # eps_c_e: tridiag + coupling to c_s surfaces
    for i in range(N_tot):
        row = sl["eps_c_e"].start + i
        for di in [-1, 0, 1]:
            if 0 <= i + di < N_tot:
                rows.append(row); cols.append(sl["eps_c_e"].start + i + di)
        # Source coupling to particle surface concentrations
        if i < N_n:
            rows.append(row); cols.append(sl["c_s_n"].start + i * N_r + N_r - 1)
        elif i >= N_n + N_s:
            ip = i - N_n - N_s
            rows.append(row); cols.append(sl["c_s_p"].start + ip * N_r + N_r - 1)

    # phi_s and phi_e rows: all zeros (dy/dt = 0), but mark diagonal for safety
    for i in range(N_n):
        row = sl["phi_s_n"].start + i
        rows.append(row); cols.append(row)
    for i in range(N_p):
        row = sl["phi_s_p"].start + i
        rows.append(row); cols.append(row)
    for i in range(N_tot):
        row = sl["phi_e"].start + i
        rows.append(row); cols.append(row)

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(len(rows))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))


# =============================================================================
# 15. MAIN SIMULATION
# =============================================================================

def run_dfn_simulation(C_rate=1.0, material="NCM",
                       recipe_CA=None, recipe_Binder=None,
                       porosity_p=None,
                       N_n=15, N_s=8, N_p=15, N_r=8,
                       t_max=None, tau=1e-3, plot=True,
                       cell_type="Chen2020"):
    """Run a full DFN discharge simulation.

    Parameters
    ----------
    C_rate : float
        Discharge C-rate (e.g. 1.0 for 1C).
    material : str
        Cathode material: 'NCM', 'LFP', or 'LMFP'.
    recipe_CA : float or None
        Conductive additive weight % in cathode recipe.
    recipe_Binder : float or None
        Binder weight % in cathode recipe.
    porosity_p : float or None
        Cathode porosity override. If None, uses default.
    """
    print("=" * 70)
    print(f"  DFN MODEL — PyBaMM-Faithful Implementation")
    print(f"  Cell: {cell_type} | Material: {material} | C-rate: {C_rate}C")
    print("=" * 70)

    # Load base cell parameters
    if cell_type == "Prada2013":
        p = load_prada2013_params()
    else:
        p = load_chen2020_params()
    mat = load_material_params(material)

    # Store active material density for specific capacity calculation [g/cm³]
    p["rho_AM"] = mat["rho_AM"]

    # For Prada2013, parameters are already set for LFP — skip material override
    if cell_type != "Prada2013":
        # Override cathode parameters from material
        p["c_s_max_p"] = mat["c_s_max_p"]
        p["c_s_init_p"] = mat["c_s_init_p"]
        p["D_s_p"] = mat["D_s_p"]
        p["R_p_p"] = mat["R_p_p"]
        p["sigma_p"] = mat["sigma_p"]
        p["k_p"] = mat["k_p"]
        p["V_min"] = mat["V_min"]
        p["V_max"] = mat["V_max"]
        p["OCP_p"] = mat["OCP_p"]

    # Capacity balance: for non-NCM materials on Chen2020 geometry,
    # match anode and cathode capacities.
    if material != "NCM" and cell_type != "Prada2013":
        sto_p_init = mat["sto_p_init"]
        sto_p_full = mat["sto_p_full"]
        # Max cathode capacity (mol/m² of electrode area)
        Q_p_max = p["eps_s_p"] * p["L_p"] * p["c_s_max_p"] * (sto_p_full - sto_p_init)
        # Max anode capacity
        sto_n_orig = p["c_s_init_n"] / p["c_s_max_n"]  # 0.9014
        sto_n_min = 0.03
        Q_n_max = p["eps_s_n"] * p["L_n"] * p["c_s_max_n"] * (sto_n_orig - sto_n_min)

        if Q_p_max < Q_n_max:
            # Cathode-limited: reduce anode initial SOC so anode capacity = cathode capacity
            delta_sto_n = Q_p_max / (p["eps_s_n"] * p["L_n"] * p["c_s_max_n"])
            sto_n_new = sto_n_min + delta_sto_n
            p["c_s_init_n"] = p["c_s_max_n"] * sto_n_new
            print(f"  Capacity balance (cathode-limited):")
            print(f"    Anode sto: {sto_n_orig:.3f} -> {sto_n_new:.3f}")
            print(f"    Cathode sto: {sto_p_init:.3f} -> {sto_p_full:.3f}")
        else:
            # Anode-limited: reduce cathode delta
            delta_sto_p = Q_n_max / (p["eps_s_p"] * p["L_p"] * p["c_s_max_p"])
            sto_p_end = sto_p_init + delta_sto_p
            print(f"  Capacity balance (anode-limited):")
            print(f"    Cathode sto: {sto_p_init:.3f} -> {sto_p_end:.3f}")

        p["c_s_init_p"] = p["c_s_max_p"] * sto_p_init

    # Apply recipe (percolation theory) if given
    if recipe_CA is not None and recipe_Binder is not None:
        wt_AM = 100.0 - recipe_CA - recipe_Binder
        rho_AM = mat["rho_AM"]
        eps_e_p = porosity_p if porosity_p is not None else p["eps_e_p"]

        props = recipe_to_electrode_props(
            wt_AM, recipe_CA, recipe_Binder, rho_AM,
            porosity=eps_e_p, R_p=p["R_p_p"]
        )
        p["eps_s_p"] = props["eps_s"]
        p["eps_e_p"] = props["eps_e"]
        p["sigma_p"] = props["sigma_eff"]
        p["sigma_eff_p"] = props["sigma_eff"]
        p["a_p"] = props["a"]

        print(f"  Recipe: AM={wt_AM:.1f}%, CA={recipe_CA:.1f}%, Binder={recipe_Binder:.1f}%")
        print(f"  -> eps_s={props['eps_s']:.4f}, eps_e={props['eps_e']:.4f}")
        print(f"  -> sigma_eff={props['sigma_eff']:.4f} S/m (percolation)")
        print(f"  -> a_p={props['a']:.0f} 1/m")
    else:
        # Default: recalculate a_p from material R_p
        p["a_p"] = 3.0 * p["eps_s_p"] / p["R_p_p"]
        p["sigma_eff_p"] = p["sigma_p"]
        if porosity_p is not None:
            p["eps_e_p"] = porosity_p

    I_app = p["Q_cell"] * C_rate
    if t_max is None:
        t_max = 3800.0 / C_rate

    print(f"  Applied current: {I_app:.2f} A")
    print(f"  Current density: {I_app / p['A_cell']:.2f} A/m^2")

    # Build mesh
    geo = build_cell_mesh(p, N_n, N_s, N_p, N_r)
    sl = build_slices(geo)
    N_tot = geo["N_tot"]

    n_dof = N_n * N_r + N_p * N_r + N_tot + N_n + N_p + N_tot
    print(f"  Total DOFs: {n_dof} (c_s={N_n*N_r+N_p*N_r}, c_e={N_tot}, "
          f"phi_s={N_n+N_p}, phi_e={N_tot})")

    # Initial conditions
    y0 = build_initial_conditions(p, geo, sl)
    print("  Solving consistent initial conditions...")
    y0 = solve_consistent_ic(p, geo, sl, y0, I_app)

    V_init = y0[sl["phi_s_p"]][-1] - y0[sl["phi_s_n"]][0]
    print(f"  Initial voltage: {V_init:.4f} V")

    # Build RHS (pseudo-transient DAE)
    rhs = build_rhs(p, geo, sl, I_app, tau=tau)

    # Build sparse Jacobian sparsity pattern for BDF efficiency
    print("  Building Jacobian sparsity pattern...")
    n = len(y0)
    sparsity = sparse.lil_matrix((n, n), dtype=np.float64)
    # Conservative: mark all potential couplings
    # c_s blocks: each particle couples to itself + phi_s + phi_e (via BV)
    for ix in range(N_n):
        r_sl = slice(sl["c_s_n"].start + ix * N_r,
                     sl["c_s_n"].start + (ix + 1) * N_r)
        sparsity[r_sl, r_sl] = 1  # internal diffusion
        sparsity[r_sl, sl["phi_s_n"].start + ix] = 1  # BV coupling
        sparsity[r_sl, sl["phi_e"].start + ix] = 1
    for ix in range(N_p):
        r_sl = slice(sl["c_s_p"].start + ix * N_r,
                     sl["c_s_p"].start + (ix + 1) * N_r)
        sparsity[r_sl, r_sl] = 1
        sparsity[r_sl, sl["phi_s_p"].start + ix] = 1
        sparsity[r_sl, sl["phi_e"].start + N_n + geo["N_s"] + ix] = 1
    # eps_c_e: couples to all c_e neighbors + phi_s + phi_e (via source)
    ce_s, ce_e = sl["eps_c_e"].start, sl["eps_c_e"].stop
    for i in range(N_tot):
        for j in range(max(0, i - 1), min(N_tot, i + 2)):
            sparsity[ce_s + i, ce_s + j] = 1
        # BV source coupling
        if i < N_n:
            sparsity[ce_s + i, sl["phi_s_n"].start + i] = 1
            sparsity[ce_s + i, sl["phi_e"].start + i] = 1
            sparsity[ce_s + i, sl["c_s_n"].start + i * N_r + N_r - 1] = 1
        elif i >= N_n + geo["N_s"]:
            ip = i - N_n - geo["N_s"]
            sparsity[ce_s + i, sl["phi_s_p"].start + ip] = 1
            sparsity[ce_s + i, sl["phi_e"].start + i] = 1
            sparsity[ce_s + i, sl["c_s_p"].start + ip * N_r + N_r - 1] = 1
    # phi_s_n: tridiagonal + BV coupling to c_s, c_e, phi_e
    ps_n_s = sl["phi_s_n"].start
    for i in range(N_n):
        for j in range(max(0, i - 1), min(N_n, i + 2)):
            sparsity[ps_n_s + i, ps_n_s + j] = 1
        sparsity[ps_n_s + i, sl["phi_e"].start + i] = 1
        sparsity[ps_n_s + i, ce_s + i] = 1
        sparsity[ps_n_s + i, sl["c_s_n"].start + i * N_r + N_r - 1] = 1
    # phi_s_p
    ps_p_s = sl["phi_s_p"].start
    for i in range(N_p):
        for j in range(max(0, i - 1), min(N_p, i + 2)):
            sparsity[ps_p_s + i, ps_p_s + j] = 1
        ip_e = N_n + geo["N_s"] + i
        sparsity[ps_p_s + i, sl["phi_e"].start + ip_e] = 1
        sparsity[ps_p_s + i, ce_s + ip_e] = 1
        sparsity[ps_p_s + i, sl["c_s_p"].start + i * N_r + N_r - 1] = 1
    # phi_e: tridiagonal + BV coupling
    pe_s = sl["phi_e"].start
    for i in range(N_tot):
        for j in range(max(0, i - 1), min(N_tot, i + 2)):
            sparsity[pe_s + i, pe_s + j] = 1
            sparsity[pe_s + i, ce_s + j] = 1  # c_e coupling via kappa, chi
        if i < N_n:
            sparsity[pe_s + i, ps_n_s + i] = 1
            sparsity[pe_s + i, sl["c_s_n"].start + i * N_r + N_r - 1] = 1
        elif i >= N_n + geo["N_s"]:
            ip = i - N_n - geo["N_s"]
            sparsity[pe_s + i, ps_p_s + ip] = 1
            sparsity[pe_s + i, sl["c_s_p"].start + ip * N_r + N_r - 1] = 1
    jac_sparsity = sparsity.tocsc()
    print(f"  Sparsity: {jac_sparsity.nnz} nonzeros / {n*n} total "
          f"({100*jac_sparsity.nnz/(n*n):.1f}%)")

    # Events: voltage cutoff + c_e depletion
    def voltage_cutoff(t, y):
        V = y[sl["phi_s_p"]][-1] - y[sl["phi_s_n"]][0]
        return V - p["V_min"]
    voltage_cutoff.terminal = True
    voltage_cutoff.direction = -1

    def ce_depletion(t, y):
        c_e_local = y[sl["eps_c_e"]] / geo["eps_e"]
        return np.min(c_e_local) - 5.0
    ce_depletion.terminal = True
    ce_depletion.direction = -1

    # Progress callback
    t_start = timer.time()
    last_print = [0.0]

    def progress(t, y):
        if t - last_print[0] >= max(60.0, t_max / 30):
            c_e_now = y[sl["eps_c_e"]] / geo["eps_e"]
            V = y[sl["phi_s_p"]][-1] - y[sl["phi_s_n"]][0]
            c_s_sn = y[sl["c_s_n"]].reshape(N_n, N_r)[:, -1]
            c_s_sp = y[sl["c_s_p"]].reshape(N_p, N_r)[:, -1]
            sto_n = c_s_sn[N_n // 2] / p["c_s_max_n"]
            sto_p = c_s_sp[N_p // 2] / p["c_s_max_p"]
            elapsed = timer.time() - t_start
            print(f"  {t:7.1f}s  V={V:.4f}  c_e=[{c_e_now.min():.1f},{c_e_now.max():.1f}]"
                  f"  sto_n={sto_n:.4f}  sto_p={sto_p:.4f}  [{elapsed:.0f}s wall]")
            last_print[0] = t
        return 0
    progress.terminal = False

    # Solve
    print(f"\n  Solving with BDF (tau={tau:.0e})...")
    print(f"  {'Time':>8s}  {'Voltage':>8s}  {'c_e range':>16s}  "
          f"{'sto_n':>8s}  {'sto_p':>8s}  {'Wall':>8s}")
    print("  " + "-" * 66)

    sol = solve_ivp(
        rhs,
        t_span=(0, t_max),
        y0=y0,
        method='BDF',
        rtol=1e-6,
        atol=1e-8,
        jac_sparsity=jac_sparsity,
        events=[voltage_cutoff, ce_depletion, progress],
        max_step=10.0,
        first_step=0.01,
    )

    elapsed = timer.time() - t_start
    t_sol = sol.t
    y_sol = sol.y

    # Extract voltage at each time step
    V_sol = y_sol[sl["phi_s_p"]][-1, :] - y_sol[sl["phi_s_n"]][0, :]

    print(f"\n  Solver status: {sol.message}")
    print(f"  Steps: {sol.t.size}, Wall time: {elapsed:.1f}s")
    print(f"  V: {V_sol[0]:.4f} -> {V_sol[-1]:.4f} V")
    print(f"  Duration: {t_sol[-1]:.0f}s ({t_sol[-1]/3600:.2f}h)")

    if sol.t_events[0].size > 0:
        print(f"  Voltage cutoff at t = {sol.t_events[0][0]:.1f}s")
    if sol.t_events[1].size > 0:
        print(f"  c_e depletion at t = {sol.t_events[1][0]:.1f}s")

    # ---- PLOTTING ----
    if not plot:
        return t_sol, V_sol, p, geo
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'DFN Model — {material} | {C_rate}C Discharge', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(t_sol / 3600, V_sol, 'b-', linewidth=2)
    ax.axhline(y=p["V_min"], color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Cell Voltage [V]')
    ax.set_title('Discharge Curve')
    ax.margins(x=0.05)
    ax.grid(True, alpha=0.3)

    # Final state
    y_final = y_sol[:, -1]
    c_e_f = np.clip(y_final[sl["eps_c_e"]] / geo["eps_e"], 1.0, 5000.0)
    phi_s_n_f = y_final[sl["phi_s_n"]]
    phi_s_p_f = y_final[sl["phi_s_p"]]
    phi_e_f = y_final[sl["phi_e"]]
    c_s_n_f = y_final[sl["c_s_n"]].reshape(N_n, N_r)
    c_s_p_f = y_final[sl["c_s_p"]].reshape(N_p, N_r)

    ax = axes[0, 1]
    x_um = geo["x_nodes"] * 1e6
    ax.plot(x_um, c_e_f, 'b-', linewidth=2)
    ax.set_xlabel('Position [um]')
    ax.set_ylabel('c_e [mol/m3]')
    ax.set_title('Final Electrolyte Concentration')
    ax.margins(x=0.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    x_n_um = geo["mesh_n"]["nodes"] * 1e6
    x_p_um = (geo["mesh_p"]["nodes"] + p["L_n"] + p["L_s"]) * 1e6
    sto_n_f = c_s_n_f[:, -1] / p["c_s_max_n"]
    sto_p_f = c_s_p_f[:, -1] / p["c_s_max_p"]
    ax.plot(x_n_um, sto_n_f, 'b-o', ms=3, label='Neg')
    ax.plot(x_p_um, sto_p_f, 'r-o', ms=3, label='Pos')
    ax.set_xlabel('Position [um]')
    ax.set_ylabel('Surface Stoichiometry')
    ax.set_title('Final Surface SOC')
    ax.margins(x=0.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_um, phi_e_f, 'g-', lw=2, label='phi_e')
    ax.plot(x_n_um, phi_s_n_f, 'b-', lw=2, label='phi_s (neg)')
    ax.plot(x_p_um, phi_s_p_f, 'r-', lw=2, label='phi_s (pos)')
    ax.set_xlabel('Position [um]')
    ax.set_ylabel('Potential [V]')
    ax.set_title('Final Potentials')
    ax.margins(x=0.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/Users/essential/Desktop/캡스톤 디자인/DFN_result.png', dpi=150)
    plt.show()

    return t_sol, V_sol, p, geo


# =============================================================================
# 16. CSV EXPORT
# =============================================================================

def export_csv(t_sol, V_sol, p, filepath, C_rate=1.0, material="NCM"):
    """Export simulation results to CSV.

    Columns: time_s, time_h, voltage_V, capacity_Ah, SOC
    """
    import csv
    Q_cell = p["Q_cell"]
    cap_Ah = t_sol * Q_cell / 3600.0   # discharged capacity [Ah]
    soc = 1.0 - cap_Ah / (Q_cell * 1.0 / C_rate * Q_cell / 3600.0)
    # simpler: SOC = 1 - (I*t)/(Q_cell) where I = Q_cell * C_rate
    soc = 1.0 - (Q_cell * C_rate * t_sol) / (Q_cell * 3600.0)
    soc = np.clip(soc, 0.0, 1.0)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "time_h", "voltage_V", "capacity_Ah", "SOC",
                         "material", "C_rate"])
        for i in range(len(t_sol)):
            writer.writerow([
                f"{t_sol[i]:.4f}",
                f"{t_sol[i]/3600:.6f}",
                f"{V_sol[i]:.6f}",
                f"{cap_Ah[i]:.6f}",
                f"{soc[i]:.6f}",
                material,
                f"{C_rate:.2f}",
            ])
    print(f"  CSV saved: {filepath} ({len(t_sol)} rows)")


def compare_crates(material="NCM", crates=None, recipe_CA=None,
                   recipe_Binder=None, porosity_p=None,
                   N_n=15, N_s=8, N_p=15, N_r=8, save_csv=True):
    """Run simulations at multiple C-rates and plot comparison.

    Parameters
    ----------
    material : str
        Cathode material ('NCM', 'LFP', 'LMFP').
    crates : list of float
        C-rates to simulate. Default: [0.5, 1.0, 2.0, 3.0].
    save_csv : bool
        If True, export each result to CSV.
    """
    if crates is None:
        crates = [0.5, 1.0, 2.0, 3.0]

    base_dir = "/Users/essential/Desktop/캡스톤 디자인"
    results = {}
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(crates)))

    for i, cr in enumerate(crates):
        print(f"\n{'='*70}")
        print(f"  C-rate: {cr}C")
        print(f"{'='*70}")
        try:
            t, V, p_out, geo = run_dfn_simulation(
                C_rate=cr, material=material,
                recipe_CA=recipe_CA, recipe_Binder=recipe_Binder,
                porosity_p=porosity_p,
                N_n=N_n, N_s=N_s, N_p=N_p, N_r=N_r, plot=False,
            )
            results[cr] = (t, V, p_out)
            if save_csv:
                csv_path = f"{base_dir}/DFN_{material}_{cr:.1f}C.csv"
                export_csv(t, V, p_out, csv_path, C_rate=cr, material=material)
        except Exception as e:
            print(f"  {cr}C failed: {e}")

    if not results:
        print("No successful simulations.")
        return results

    # --- Plot 1: Voltage vs Time ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"DFN: {material} — C-rate Comparison", fontsize=14, fontweight='bold')

    ax = axes[0]
    for i, (cr, (t, V, p_out)) in enumerate(sorted(results.items())):
        ax.plot(t / 3600, V, color=colors[i], linewidth=2, label=f"{cr}C")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Voltage vs Time")
    ax.margins(x=0.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Voltage vs Specific Capacity ---
    ax = axes[1]
    for i, (cr, (t, V, p_out)) in enumerate(sorted(results.items())):
        cap_Ah = t * p_out["Q_cell"] * cr / 3600.0
        mass_AM_g = p_out["eps_s_p"] * p_out["L_p"] * p_out["A_cell"] * p_out["rho_AM"] * 1e6
        cap = cap_Ah * 1000.0 / mass_AM_g
        ax.plot(cap, V, color=colors[i], linewidth=2, label=f"{cr}C")
    ax.set_xlabel("Specific Capacity [mAh/g]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Voltage vs Specific Capacity")
    ax.set_xlim(0, 240)
    ax.set_ylim(0, 4.5)
    ax.set_xticks(np.arange(0, 241, 20))
    ax.set_yticks(np.arange(0, 4.51, 0.5))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Delivered capacity bar chart ---
    ax = axes[2]
    crs_sorted = sorted(results.keys())
    delivered = []
    for cr in crs_sorted:
        t, V, p_out = results[cr]
        cap_Ah = t[-1] * p_out["Q_cell"] * cr / 3600.0
        mass_AM_g = p_out["eps_s_p"] * p_out["L_p"] * p_out["A_cell"] * p_out["rho_AM"] * 1e6
        delivered.append(cap_Ah * 1000.0 / mass_AM_g)
    bar_colors = [colors[i] for i in range(len(crs_sorted))]
    bars = ax.bar([f"{cr}C" for cr in crs_sorted], delivered, color=bar_colors)
    for bar, cap in zip(bars, delivered):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{cap:.1f}", ha='center', va='bottom', fontsize=10)
    ax.set_xlabel("C-rate")
    ax.set_ylabel("Delivered Specific Capacity [mAh/g]")
    ax.set_title("Specific Capacity vs C-rate")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = f"{base_dir}/DFN_{material}_Crate_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"\n  Plot saved: {plot_path}")

    return results


# =============================================================================
# 17. ENTRY POINT (CLI)
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DFN Battery Discharge Simulator (PyBaMM-faithful)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", default="run",
                        choices=["run", "compare_materials", "compare_recipes",
                                 "compare_crates"],
                        help=("Simulation mode:\n"
                              "  run              — Single discharge (default)\n"
                              "  compare_materials — NCM vs LFP vs LMFP\n"
                              "  compare_recipes  — Recipe effect comparison\n"
                              "  compare_crates   — C-rate comparison"))
    parser.add_argument("-m", "--material", default="NCM",
                        choices=["NCM", "LFP", "LMFP", "LMFP64", "LMFP73", "LMFP82"],
                        help="Cathode material (default: NCM)")
    parser.add_argument("-c", "--crate", type=float, default=1.0,
                        help="C-rate for single run (default: 1.0)")
    parser.add_argument("--crates", type=str, default=None,
                        help="Comma-separated C-rates for compare_crates "
                             "(e.g. '0.2,0.5,1,2,3,5')")
    parser.add_argument("--ca", type=float, default=None,
                        help="Conductive additive wt%% in recipe")
    parser.add_argument("--binder", type=float, default=None,
                        help="Binder wt%% in recipe")
    parser.add_argument("--porosity", type=float, default=None,
                        help="Cathode porosity override")
    parser.add_argument("--csv", action="store_true",
                        help="Export results to CSV")
    parser.add_argument("--no-plot", action="store_true",
                        help="Suppress plots")
    args = parser.parse_args()

    base_dir = "/Users/essential/Desktop/캡스톤 디자인"

    if args.mode == "run":
        # --- Single discharge simulation ---
        t_sol, V_sol, p, geo = run_dfn_simulation(
            C_rate=args.crate, material=args.material,
            recipe_CA=args.ca, recipe_Binder=args.binder,
            porosity_p=args.porosity,
            N_n=15, N_s=8, N_p=15, N_r=8,
            plot=not args.no_plot,
        )
        if args.csv:
            csv_path = f"{base_dir}/DFN_{args.material}_{args.crate:.1f}C.csv"
            export_csv(t_sol, V_sol, p, csv_path,
                       C_rate=args.crate, material=args.material)

    elif args.mode == "compare_crates":
        # --- C-rate comparison ---
        if args.crates:
            crates = [float(x.strip()) for x in args.crates.split(",")]
        else:
            crates = [0.5, 1.0, 2.0, 3.0]
        compare_crates(
            material=args.material, crates=crates,
            recipe_CA=args.ca, recipe_Binder=args.binder,
            porosity_p=args.porosity,
            save_csv=args.csv,
        )

    elif args.mode == "compare_materials":
        # --- Material comparison ---
        fig, ax = plt.subplots(figsize=(10, 6))
        cr = args.crate
        for mat, color in [("NCM", "blue"), ("LFP", "green"), ("LMFP", "orange")]:
            try:
                t, V, p_m, _ = run_dfn_simulation(
                    C_rate=cr, material=mat,
                    N_n=15, N_s=8, N_p=15, N_r=8, plot=False,
                )
                cap_Ah = t * p_m["Q_cell"] * cr / 3600
                mass_AM_g = p_m["eps_s_p"] * p_m["L_p"] * p_m["A_cell"] * p_m["rho_AM"] * 1e6
                cap = cap_Ah * 1000.0 / mass_AM_g
                ax.plot(cap, V, color=color, linewidth=2, label=mat)
                if args.csv:
                    export_csv(t, V, p_m,
                               f"{base_dir}/DFN_{mat}_{cr:.1f}C.csv",
                               C_rate=cr, material=mat)
            except Exception as e:
                print(f"  {mat} failed: {e}")
        ax.set_xlabel("Specific Capacity [mAh/g]")
        ax.set_ylabel("Voltage [V]")
        ax.set_title(f"DFN: Material Comparison ({cr}C Discharge)")
        ax.set_xlim(0, 240)
        ax.set_ylim(0, 4.5)
        ax.set_xticks(np.arange(0, 241, 20))
        ax.set_yticks(np.arange(0, 4.51, 0.5))
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{base_dir}/DFN_material_comparison.png", dpi=150)
        if not args.no_plot:
            plt.show()

    elif args.mode == "compare_recipes":
        # --- Recipe comparison ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        cr = args.crate
        recipes = [
            (96.0, 2.0, 2.0, "96/2/2"),
            (94.0, 3.0, 3.0, "94/3/3"),
            (90.0, 5.0, 5.0, "90/5/5"),
            (80.0, 10.0, 10.0, "80/10/10"),
        ]
        for wt_AM, wt_CA, wt_B, label in recipes:
            try:
                t, V, p_r, _ = run_dfn_simulation(
                    C_rate=cr, material=args.material,
                    recipe_CA=wt_CA, recipe_Binder=wt_B,
                    N_n=15, N_s=8, N_p=15, N_r=8, plot=False,
                )
                cap_Ah = t * p_r["Q_cell"] * cr / 3600
                mass_AM_g = p_r["eps_s_p"] * p_r["L_p"] * p_r["A_cell"] * p_r["rho_AM"] * 1e6
                cap = cap_Ah * 1000.0 / mass_AM_g
                ax1.plot(cap, V, linewidth=2, label=f"AM/CA/B={label}")
                if args.csv:
                    export_csv(t, V, p_r,
                               f"{base_dir}/DFN_{args.material}_recipe_{label.replace('/','-')}_{cr:.1f}C.csv",
                               C_rate=cr, material=args.material)
            except Exception as e:
                print(f"  Recipe {label} failed: {e}")
        ax1.set_xlabel("Specific Capacity [mAh/g]")
        ax1.set_ylabel("Voltage [V]")
        ax1.set_title(f"DFN: Recipe Effect ({args.material}, {cr}C)")
        ax1.set_xlim(0, 240)
        ax1.set_ylim(0, 4.5)
        ax1.set_xticks(np.arange(0, 241, 20))
        ax1.set_yticks(np.arange(0, 4.51, 0.5))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        mat_params = load_material_params(args.material)
        sigmas, labels = [], []
        for wt_AM, wt_CA, wt_B, label in recipes:
            props = recipe_to_electrode_props(
                wt_AM, wt_CA, wt_B, rho_AM=mat_params["rho_AM"],
                porosity=0.335, R_p=mat_params["R_p_p"],
            )
            sigmas.append(props["sigma_eff"])
            labels.append(label)
        ax2.bar(labels, sigmas, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"])
        ax2.set_xlabel("AM/CA/Binder wt%")
        ax2.set_ylabel("sigma_eff [S/m]")
        ax2.set_title("Cathode Conductivity (Percolation Theory)")
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f"{base_dir}/DFN_recipe_comparison.png", dpi=150)
        if not args.no_plot:
            plt.show()
