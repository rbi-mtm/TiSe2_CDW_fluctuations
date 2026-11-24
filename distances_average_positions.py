import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]
species_name = "Ti"

ref = read('./data/SPOSCAR_cartesian')
ref_pos = ref.get_positions()
ti_idx = [i for i, a in enumerate(ref) if a.symbol == species_name]
ref_ti_pos = ref_pos[ti_idx]

global_min = np.inf
global_max = -np.inf

for temp in temps:
    traj_file = f"./bussi_md/{temp}/bussi.traj"
    traj = read(traj_file, index="10000:110000")

    pos_accum = np.zeros((len(ti_idx), 3))
    n_frames = 0

    for atoms in traj:
        pos_accum += atoms.get_positions()[ti_idx]
        n_frames += 1

    avg_pos = pos_accum / n_frames
    disp = np.linalg.norm((avg_pos - ref_ti_pos)[:, :2], axis=1)

    global_min = min(global_min, disp.min())
    global_max = max(global_max, disp.max())

print('global_min:', global_min, flush=True)
print('global_max:', global_max, flush=True)

for temp in temps:
    traj_file = f"./bussi_md/{temp}/bussi.traj"
    traj = read(traj_file, index="10000:110000")

    pos_accum = np.zeros((len(ti_idx), 3))
    n_frames = 0

    for atoms in traj:
        pos_accum += atoms.get_positions()[ti_idx]
        n_frames += 1

    avg_pos = pos_accum / n_frames
    disp = np.linalg.norm((avg_pos - ref_ti_pos)[:, :2], axis=1)

    xy = ref_ti_pos[:, :2]

    plt.figure(figsize=(6, 5), dpi=300)
    plt.scatter(
        xy[:, 0], xy[:, 1],
        c=disp,
        cmap='hot_r',
        s=1.5,
        vmin=global_min,
        vmax=global_max
    )
    plt.xlabel('x (Å)')
    plt.ylabel('y (Å)')
    plt.title(f'Ti Displacements at {temp} K')
    cbar = plt.colorbar()
    cbar.set_label(r'$\left|\langle \mathbf{R}_{i} \rangle - \mathbf{R}^{0}_{i}\right|$')
    plt.tight_layout()
    plt.savefig(f'./ti_disp_{temp}K.png')
    plt.close()
