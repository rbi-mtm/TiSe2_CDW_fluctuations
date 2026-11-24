import argparse
import glob
import os
import re
import numpy as np
from ase.io import read

# ==============================================================================
# CONFIGURATION & USER PARAMETERS
# ==============================================================================
# Supercell dimensions (56 x 56 x 1)
cell_x = 56
cell_y = 56
cell_z = 1

sigma_smoothing = 4  # Real-space Gaussian smoothing width \sigma (Angstroms) -- TEST RUN
dimension = 2  # Spatial dimensions (2D order parameter field)

DEFAULT_TEMPS = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

# ------------------------------------------------------------------------------
# Command-line interface: run with the default temperature list, or pass one
# (or more) temperatures explicitly, e.g. `python3 phase_gaussian_sigma4.py 100` or
# `python3 phase_gaussian_sigma4.py 100 200 300`.
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Compute the CDW order parameter (amplitude & phase) via Gaussian smoothing."
)
parser.add_argument(
    "temps",
    nargs="*",
    type=int,
    default=None,
    help="Temperature(s) in K to process. If omitted, the built-in default list of 24 temperatures is used.",
)
args = parser.parse_args()

temps = args.temps if args.temps else DEFAULT_TEMPS

data_dir = "./order_parameter_gaussian_sigma4"
os.makedirs(data_dir, exist_ok=True)

# Reference supercell file path
sposcar_file = "./data/SPOSCAR_cartesian"

# Q-point indices, loaded from eigvec_Q{q}.npz (Ordering per file: [Se1, Se2, Ti])
q_points = [0, 1, 2]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def build_gaussian_kernel(R0, cell, sigma, d=2):
    """
    Computes Gaussian real-space kernel matrix G_{ij} with exact periodic boundary
    conditions for arbitrary (orthogonal or non-orthogonal/hexagonal) cell geometries.
    """
    cell_matrix = np.asarray(cell)  # (3, 3) matrix of lattice vectors as rows
    inv_cell = np.linalg.inv(cell_matrix)

    # 1. Convert Cartesian positions to Fractional coordinates: S = R0 @ inv(cell)
    S0 = R0 @ inv_cell  # Shape: (N_super, 3)

    # 2. Compute pairwise fractional differences
    diff_frac = S0[:, None, :] - S0[None, :, :]  # Shape: (N_super, N_super, 3)

    # 3. Apply Minimum Image Convention in fractional space (wrap to [-0.5, 0.5])
    diff_frac -= np.round(diff_frac)

    # 4. Convert fractional differences back to Cartesian vectors
    diff_cart = diff_frac @ cell_matrix  # Shape: (N_super, N_super, 3)

    # 5. Compute squared distances (in 2D x-y plane if d=2)
    if d == 2:
        dist_sq = np.sum(diff_cart[:, :, :2] ** 2, axis=-1)
    else:
        dist_sq = np.sum(diff_cart ** 2, axis=-1)

    # 6. Evaluate normalized Gaussian kernel
    norm_factor = 1.0 / ((2.0 * np.pi * sigma**2) ** (d / 2.0))
    G = norm_factor * np.exp(-dist_sq / (2.0 * sigma**2))

    return G


# ==============================================================================
# ONE-TIME GLOBAL PRE-COMPUTATIONS (R0, Q, G_ij, phase for 3 Q-points)
# ==============================================================================
print("==================================================================", flush=True)
print(" Loading Reference Structure & Building Global Kernel Matrix", flush=True)
print("==================================================================", flush=True)

# Read reference positions and cell parameters from SPOSCAR_cartesian ONCE
ref_structure = read(sposcar_file)
R0 = ref_structure.get_positions()
cell = ref_structure.cell  # ASE Cell object

# Setup Supercell Mapping (sublattices)
N_unit_cells = cell_x * cell_y * cell_z  # 3136 unit cells
sublattices = np.repeat([0, 1, 2], N_unit_cells)
N_super = len(sublattices)  # 9408 atoms total

# Pre-compute reciprocal lattice
reciprocal_cell_super = 2.0 * np.pi * np.linalg.inv(cell).T

# Load Q-point eigenvectors from eigvec_Q{q}.npz files
m_points = {}
for q in q_points:
    eigvec_file = f"./data/eigvec_Q{q}.npz"
    eig_data = np.load(eigvec_file)
    # Guard: the eigenvectors must be the complex ones written by eigenvec_3.py
    # (one global phase, per-atom phases kept). eigenvec_2.py writes real vectors
    # to the same file names, which give Se the wrong phase.
    if not np.iscomplexobj(eig_data["e_alpha_j"]):
        raise SystemExit(f"{eigvec_file} holds real eigenvectors (from eigenvec_2.py); rerun eigenvec_3.py first.")
    m_points[q] = {
        "Q_frac": eig_data["Q_alpha"],
        "e_primitive": eig_data["e_alpha_j"],
    }

E_super = {}
Q_cartesian = {}
carrier_phase = {}

for q in q_points:
    # Map primitive mode vectors to every site in the supercell for Q-point q
    E_super[q] = m_points[q]["e_primitive"][sublattices]  # Shape: (N_super, 3)

    # Calculate Q_cartesian (in 1/Angstrom) using supercell reciprocal lattice
    Q_frac_super = m_points[q]["Q_frac"] * np.array([cell_x, cell_y, cell_z])
    Q_cartesian[q] = Q_frac_super @ reciprocal_cell_super

    # Precalculate complex phase factor exp(-i * Q_q . R_j^0)
    carrier_phase[q] = np.exp(-1j * np.dot(R0, Q_cartesian[q]))

# Build Gaussian kernel matrix G_ij ONCE with exact fractional PBC wrapping
print(f"Building Gaussian smoothing matrix G_ij for {N_super} atoms...", flush=True)
G = build_gaussian_kernel(R0, cell=cell, sigma=sigma_smoothing, d=dimension)
# Row-normalize: divide by N_i = sum_j G_ij so that Psi_i is the Gaussian-weighted AVERAGE
# displacement (units A, same as phase_no_gaussian.py). Without this, Psi is a displacement
# density in 1/A (N_i = site density, 0.277 1/A^2 for sigma = 2 A). Files written by this
# version carry kernel_row_normalized = True, so the plotting scripts do not rescale them again;
# older data_3 files (without the flag) are rescaled by 1/N_i in the plotting scripts.
G /= G.sum(axis=1, keepdims=True)
print("Global setup complete! Starting temperature loop...\n", flush=True)


# ==============================================================================
# SCRIPT EXECUTION (TEMPERATURE LOOP)
# ==============================================================================
results_summary = {}

for temp in temps:
    print(f"==================================================================", flush=True)
    print(f" Processing Temperature: {temp}K", flush=True)
    print(f"==================================================================", flush=True)

    # Read trajectory frames using ASE
    traj_file = f"./bussi_md/{temp}/bussi.traj"
    frames = read(traj_file, index="10000:110000:5")
    N_frames = len(frames)

    # Initialize storage dictionary for all 3 Q-points
    Psi_t = {q: np.zeros((N_frames, N_super), dtype=np.complex128) for q in q_points}

    print("Started loop over timesteps!", flush=True)

    # Process each frame across MD trajectory
    for t in range(N_frames):
        # Displacement field u_j(t) = R_j(t) - R_j^0
        u_t = frames[t].get_positions() - R0

        for q in q_points:
            # Step 2: Project onto mode eigenvector s_j(t) = e^*_{\alpha, j} . u_j(t)
            s_t = np.sum(np.conj(E_super[q]) * u_t, axis=1)

            # Step 3: Demodulate carrier wave \zeta_j(t) = s_j(t) * exp(-i * Q_\alpha . R_j^0)
            zeta_t = s_t * carrier_phase[q]

            # Step 4: Apply precomputed Gaussian smoothing matrix G_ij
            Psi_t[q][t] = np.dot(G, zeta_t)

    print("Finished loop over timesteps!", flush=True)

    # --------------------------------------------------------------------------
    # TIME-AVERAGING AND SAVING RESULTS
    # --------------------------------------------------------------------------
    # Average the complex order parameter first, then decompose it as Psi = A*e^{i*phi}
    temp_summary = {}
    save_dict = {"R0": R0, "kernel_row_normalized": True, "sigma_smoothing": sigma_smoothing}

    for q in q_points:
        Psi_bar = np.mean(Psi_t[q], axis=0)
        A_bar = np.abs(Psi_bar)
        phi_bar = np.angle(Psi_bar)

        # Spatial averages across supercell sites: average the complex field
        # first (amplitude-weighted circular average), then decompose -- phi_bar
        # is a circular quantity wrapped into (-pi, pi], so a plain np.mean over
        # sites is not well-defined when the phase distribution straddles the
        # +-pi branch cut (which it does in practice).
        Psi_bar_spatial = np.mean(Psi_bar)
        mean_A = np.abs(Psi_bar_spatial)
        mean_phi = np.angle(Psi_bar_spatial)

        print(f"\n  [Q_point {q}]: <A_bar_{q}> = {mean_A:.4f} Ang | <phi_bar_{q}> = {mean_phi:.4f} rad", flush=True)

        temp_summary[f"A_{q}"] = mean_A
        temp_summary[f"phi_{q}"] = mean_phi

        # Collect arrays with index q for saving
        save_dict[f"Psi_t_{q}"] = Psi_t[q]
        save_dict[f"Psi_bar_{q}"] = Psi_bar
        save_dict[f"A_bar_{q}"] = A_bar
        save_dict[f"phi_bar_{q}"] = phi_bar

    # Save all indexed arrays into npz file
    out_file = os.path.join(data_dir, f"cdw_order_parameter_{temp}K.npz")
    np.savez_compressed(out_file, **save_dict)
    print(f"\n  Saved all 3 Q-point arrays into npz: {out_file}", flush=True)

    results_summary[temp] = temp_summary

# ==============================================================================
# SUMMARY TABLE ACROSS ALL TEMPERATURES & Q-POINTS
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("CDW ORDER PARAMETER TEMPERATURE SUMMARY ACROSS ALL 3 Q-POINTS", flush=True)
print("=" * 70, flush=True)
header = f"{'Temp':<6} | {'Q-pt':<5} | {'<A_bar>':<16} | {'<phi_bar>':<16}"
print(header, flush=True)
print("-" * 70, flush=True)

for temp, val in results_summary.items():
    for q in q_points:
        t_str = f"{temp}K" if q == q_points[0] else ""
        A = val[f"A_{q}"]
        phi = val[f"phi_{q}"]
        print(f"{t_str:<6} | {q:<5} | {A:<16.4f} | {phi:<16.4f}", flush=True)
    print("-" * 70, flush=True)
print("=" * 70, flush=True)
