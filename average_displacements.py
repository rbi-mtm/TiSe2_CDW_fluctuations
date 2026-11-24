from ase.io import read
import numpy as np
from ase.build import make_supercell
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

slices = [
    '10000:30000',
    '30000:50000',
    '50000:70000',
    '70000:90000',
    '90000:110000'
]

atoms_unit = read('./data/c2db-4732.json')
unit_cell = atoms_unit.get_cell()
lx, ly, lz = unit_cell.lengths()

M = [[2,0,0],[0,2,0],[0,0,1]]
supercell = make_supercell(atoms_unit, M)
supercell_cell = supercell.get_cell()

symbols_supercell = supercell.get_chemical_symbols()
ti_indices_supercell = [i for i, s in enumerate(symbols_supercell) if s == 'Ti']

R_ti_supercell = supercell.get_positions(wrap=False)[ti_indices_supercell]

R_ti_supercell[:,0] += lx/2 - ly*np.sqrt(3)/4
R_ti_supercell[:,1] += ly/2

ref_pos_xy = R_ti_supercell[:, :2]
n_ref_points = len(ref_pos_xy)

all_slice_means = []

for slice_range in slices:
    print(f"Processing slice: {slice_range}", flush=True)

    frame_means = []

    for temp in temps:
        print(f"  Temperature: {temp}", flush=True)

        traj = read(
            f'./bussi_md/{temp}/bussi.traj',
            index=slice_range
        )

        temp_frame_means = []

        for atom in traj:
            atom.positions[:,0] += lx/2 - ly*np.sqrt(3)/4
            atom.positions[:,1] += ly/2
            atom.cell = supercell_cell
            atom.wrap()

            symbols = atom.get_chemical_symbols()
            ti_indices = [i for i, s in enumerate(symbols) if s == 'Ti']
            R = atom.get_positions(wrap=False)[ti_indices]

            frame_pos_xy = R[:, :2]

            distances = cdist(frame_pos_xy, ref_pos_xy)
            closest_ref = np.argmin(distances, axis=1)

            frame_group_means = []
            for ref_idx in range(n_ref_points):
                group_positions = frame_pos_xy[closest_ref == ref_idx]
                group_mean = np.mean(group_positions, axis=0)
                frame_group_means.append(group_mean)

            temp_frame_means.append(frame_group_means)

        frame_means.append(np.array(temp_frame_means))

    slice_means = [np.mean(frame_means[i], axis=0) for i in range(len(temps))]
    all_slice_means.append(np.array(slice_means))

all_slice_means = np.array(all_slice_means)  # [n_slices, n_temps, 4, 2]

grouped_means_method1 = np.mean(all_slice_means, axis=0)  # [n_temps, 4, 2]

dists = np.linalg.norm(grouped_means_method1 - R_ti_supercell[:, :2], axis=2)

adjusted_means = np.zeros_like(grouped_means_method1)

for i in range(len(temps)):
    min_idx = np.argmin(dists[i])
    shift_vector = grouped_means_method1[i, min_idx] - R_ti_supercell[min_idx, :2]
    adjusted_means[i] = grouped_means_method1[i] - shift_vector

adjusted_dists = np.linalg.norm(adjusted_means - R_ti_supercell[:, :2], axis=2)

hlp_adjusted = adjusted_dists - np.min(adjusted_dists, axis=1)[:, None]

average_distances = np.array([np.mean(np.delete(hlp_adjusted[i], np.argmin(hlp_adjusted[i]))) for i in range(len(temps))])

plt.figure(figsize=(5,4), dpi=200)

plt.plot(temps, average_distances, marker='o', linewidth=1, markersize=4)

plt.xlabel('Temperature (K)')
plt.ylabel(r'$\delta$ Ti (Å)')

plt.axhline(0, linestyle=':', linewidth=1)

plt.tight_layout()

plt.savefig('./ti_average_displacements.png')
plt.close()
