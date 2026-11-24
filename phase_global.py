
import argparse
import os
import numpy as np
from ase.io import read

# ==============================================================================
# GLOBAL (SUPERCELL-AVERAGED) CDW ORDER PARAMETER
# ==============================================================================
# Same projection and demodulation as phase_gaussian_3.py / phase_no_gaussian.py,
# but instead of a local field Psi_i(t) (one value per atom), everything is summed
# over the whole supercell:
#
#   Psi_alpha(t) = (1/N) sum_j  [e*_{alpha,j} . u_j(t)] * exp(-i Q_alpha . R_j^0)
#
# with N = 9408 atoms. This is the sigma -> infinity limit of the row-normalised
# Gaussian kernel (G_ij -> 1/N for every i, j): Psi no longer depends on R_i^0,
# there is ONE complex number per M point per frame. Units: Angstrom (average
# projected displacement), the same as phase_no_gaussian.py and the row-normalised
# phase_gaussian_*.py runs, so the numbers can be compared directly.
#
# A uniform translation u_j = c drops out exactly: at an M point
# sum over one sublattice of exp(-i Q . R_j^0) = 0.
#
# The sublattice-resolved averages (Se1, Se2, Ti; each over 3136 atoms) are saved
# as well:  Psi_alpha(t) = mean of the three, so a sign disagreement between Ti
# and Se (cf. the sigma = 2 A stripes) can be checked directly.
#
# Because there is no time-independent spatial structure left, the second-method
# amplitude <|Psi(t)|>_t and the moments <Psi^2>_t, <Psi^4>_t are computed here
# directly (no separate amplitudes_second_method script is needed).
# ==============================================================================
cell_x = 56
cell_y = 56
cell_z = 1

DEFAULT_TEMPS = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

# MD bookkeeping (./bussi_md/{T}/bussi.py): every step written, dt = 1.5 fs.
MD_TIMESTEP_FS = 1.5
FRAME_START, FRAME_STOP, FRAME_STRIDE = 10000, 110000, 5

parser = argparse.ArgumentParser(
    description="Compute the supercell-averaged (global) CDW order parameter Psi_alpha(t)."
)
parser.add_argument("temps", nargs="*", type=int, default=None,
                    help="Temperature(s) in K. If omitted, the default list of 24 temperatures is used.")
args = parser.parse_args()
temps = args.temps if args.temps else DEFAULT_TEMPS

data_dir = "./order_parameter_global"
os.makedirs(data_dir, exist_ok=True)

sposcar_file = "./data/SPOSCAR_cartesian"

# Q-point indices, loaded from eigvec_Q{q}.npz (ordering per file: [Se1, Se2, Ti])
q_points = [0, 1, 2]
sublattice_names = ["Se1", "Se2", "Ti"]

# ==============================================================================
# ONE-TIME GLOBAL PRE-COMPUTATIONS
# ==============================================================================
print("==================================================================", flush=True)
print(" Loading Reference Structure", flush=True)
print("==================================================================", flush=True)

ref_structure = read(sposcar_file)
R0 = ref_structure.get_positions()
cell = ref_structure.cell

N_unit_cells = cell_x * cell_y * cell_z  # 3136
sublattices = np.repeat([0, 1, 2], N_unit_cells)
N_super = len(sublattices)  # 9408

reciprocal_cell_super = 2.0 * np.pi * np.linalg.inv(cell).T

m_points = {}
for q in q_points:
    eigvec_file = f"./data/eigvec_Q{q}.npz"
    eig_data = np.load(eigvec_file)
    if not np.iscomplexobj(eig_data["e_alpha_j"]):
        raise SystemExit(f"{eigvec_file} holds real eigenvectors (from eigenvec_2.py); rerun eigenvec_3.py first.")
    m_points[q] = {"Q_frac": eig_data["Q_alpha"], "e_primitive": eig_data["e_alpha_j"]}

# Projection weights w_j = e*_{alpha,j} exp(-i Q . R_j^0) / N, so that Psi(t) = sum_j w_j . u_j(t)
weights = {}
for q in q_points:
    E_super = m_points[q]["e_primitive"][sublattices]  # (N_super, 3)
    Q_frac_super = m_points[q]["Q_frac"] * np.array([cell_x, cell_y, cell_z])
    Q_cart = Q_frac_super @ reciprocal_cell_super
    carrier = np.exp(-1j * np.dot(R0, Q_cart))
    weights[q] = np.conj(E_super) * carrier[:, None]  # (N_super, 3), not yet divided by N

    # Check: uniform translation must not contribute (sum of carrier over each sublattice = 0)
    for s in range(3):
        leak = np.abs(np.sum(carrier[sublattices == s])) / N_unit_cells
        if leak > 1e-8:
            print(f"  WARNING Q{q} sublattice {sublattice_names[s]}: |mean carrier| = {leak:.2e}", flush=True)

print("Global setup complete! Starting temperature loop...\n", flush=True)

# ==============================================================================
# TEMPERATURE LOOP
# ==============================================================================
results_summary = {}

for temp in temps:
    print("==================================================================", flush=True)
    print(f" Processing Temperature: {temp}K", flush=True)
    print("==================================================================", flush=True)

    traj_file = f"./bussi_md/{temp}/bussi.traj"
    frames = read(traj_file, index=f"{FRAME_START}:{FRAME_STOP}:{FRAME_STRIDE}")
    N_frames = len(frames)
    time_ps = (FRAME_START + FRAME_STRIDE * np.arange(N_frames)) * MD_TIMESTEP_FS * 1e-3

    Psi_t = {q: np.zeros(N_frames, dtype=np.complex128) for q in q_points}
    Psi_sub_t = {q: np.zeros((N_frames, 3), dtype=np.complex128) for q in q_points}

    print("Started loop over timesteps!", flush=True)
    for t in range(N_frames):
        u_t = frames[t].get_positions() - R0
        for q in q_points:
            s_t = np.sum(weights[q] * u_t, axis=1)  # (N_super,) demodulated per-atom projection
            sub = s_t.reshape(3, N_unit_cells).mean(axis=1)  # per-sublattice average
            Psi_sub_t[q][t] = sub
            Psi_t[q][t] = sub.mean()  # = (1/N) sum_j over all atoms (equal sublattice sizes)
    print("Finished loop over timesteps!", flush=True)
    del frames

    temp_summary = {}
    save_dict = {
        "time_ps": time_ps,
        "N_atoms": N_super,
        "md_timestep_fs": MD_TIMESTEP_FS,
        "frame_slice": np.array([FRAME_START, FRAME_STOP, FRAME_STRIDE]),
        "sublattice_names": np.array(sublattice_names),
    }

    for q in q_points:
        P = Psi_t[q]
        Psi_bar = np.mean(P)
        A_bar = np.abs(Psi_bar)                   # |<Psi>_t|
        phi_bar = np.angle(Psi_bar)
        A_second = np.mean(np.abs(P))             # <|Psi|>_t
        Psi2 = np.mean(np.abs(P) ** 2)            # <|Psi|^2>_t
        Psi4 = np.mean(np.abs(P) ** 4)            # <|Psi|^4>_t
        max_im = np.max(np.abs(P.imag))

        print(f"  [Q_point {q}]: |<Psi>| = {A_bar:.4e} A | <|Psi|> = {A_second:.4e} A | "
              f"phi_bar = {phi_bar:+.4f} rad | max|Im Psi| = {max_im:.1e} A", flush=True)

        temp_summary[f"A_{q}"] = A_bar
        temp_summary[f"A2_{q}"] = A_second
        temp_summary[f"phi_{q}"] = phi_bar

        save_dict[f"Psi_t_{q}"] = P
        save_dict[f"Psi_sub_t_{q}"] = Psi_sub_t[q]
        save_dict[f"Psi_bar_{q}"] = Psi_bar
        save_dict[f"A_bar_{q}"] = A_bar
        save_dict[f"phi_bar_{q}"] = phi_bar
        save_dict[f"A_bar_second_method_{q}"] = A_second
        save_dict[f"Psi2_mean_{q}"] = Psi2
        save_dict[f"Psi4_mean_{q}"] = Psi4

    out_file = os.path.join(data_dir, f"cdw_order_parameter_global_{temp}K.npz")
    np.savez_compressed(out_file, **save_dict)
    print(f"\n  Saved: {out_file}", flush=True)
    results_summary[temp] = temp_summary

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("\n" + "=" * 78, flush=True)
print("GLOBAL CDW ORDER PARAMETER SUMMARY", flush=True)
print("=" * 78, flush=True)
print(f"{'Temp':<6} | {'Q-pt':<5} | {'|<Psi>| (A)':<14} | {'<|Psi|> (A)':<14} | {'phi_bar':<10}", flush=True)
print("-" * 78, flush=True)
for temp, val in results_summary.items():
    for q in q_points:
        t_str = f"{temp}K" if q == q_points[0] else ""
        print(f"{t_str:<6} | {q:<5} | {val[f'A_{q}']:<14.4e} | {val[f'A2_{q}']:<14.4e} | {val[f'phi_{q}']:<+10.4f}", flush=True)
    print("-" * 78, flush=True)
print("=" * 78, flush=True)
