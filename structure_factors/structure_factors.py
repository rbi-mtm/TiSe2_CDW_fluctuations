import matplotlib
matplotlib.use("Agg")
from ase.io import read
import elphmod
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
from matplotlib import colors
import torch

N_cells = [8, 56]
q_indices = [32, 1568]
ms_sizes = [400, 10]

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]
atoms_tot = [(8*7)**2 * 3, 56**2 * 3]

global_start = 10000
global_end = 110000

chunk_size = 5000
n_steps_total = global_end - global_start
n_chunks = n_steps_total // chunk_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device, flush=True)

def extract_positions(trajfile, indices=':'):
    R_all = []
    traj = read(trajfile, index=indices)

    for atoms in traj:
        R = atoms.get_positions(wrap=False) #wrap=True
        R_all.append(R)

    return np.array(R_all)

def compute_structure_factor_torch(positions_np, q_np, natoms):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    positions = torch.tensor(positions_np, dtype=torch.complex128, device=device)
    q = torch.tensor(q_np, dtype=torch.complex128, device=device)

    N_steps = positions.shape[0]
    cell = q.shape[0]
    n = natoms

    S = torch.zeros((N_steps, cell, cell), dtype=torch.complex128, device=device)
    ones = torch.ones(positions.shape[1], dtype=torch.complex128, device=device)

    for nk1 in range(cell):
        for nk2 in range(cell):
            qvec = q[nk1, nk2]  # shape (3,)

            dot = (positions.real @ qvec.real) + 1j * (positions.real @ qvec.imag)

            factor_full = torch.exp(-1j * dot)

            Sq = torch.einsum("n, tn -> t", ones, factor_full)

            Sq = (Sq.abs() ** 2) / (n ** 2)

            S[:, nk1, nk2] = Sq

    return S.cpu().numpy()

for i, cell in enumerate(N_cells):
    q_index = q_indices[i]
    S_q, S_all_q = [], []
    natoms = atoms_tot[i] #cell**2*3
    q = np.load('./k_and_q/q_voronoi_{}x{}x1.npy'.format(cell,cell))

    for j, temp in enumerate(temps):
        traj_path = '../bussi_md/{}/bussi.traj'.format(temp)
        print('{}x{}x1, {}K'.format(cell,cell,temp), flush=True)

        S_sum = None
        n_steps_accum = 0

        for c in range(n_chunks):
            start = c * chunk_size + global_start
            end = (c + 1) * chunk_size + global_start

            print('  chunk {}/{}: steps {}–{}'.format(c+1,n_chunks,start,end), flush=True)

            positions = extract_positions(traj_path, indices=slice(start, end))

            S_chunk = compute_structure_factor_torch(positions, q, natoms)

            S_chunk_sum = S_chunk.sum(axis=0)

            if S_sum is None:
                S_sum = S_chunk_sum
            else:
                S_sum += S_chunk_sum

            n_steps_accum += S_chunk.shape[0]

            del positions, S_chunk, S_chunk_sum
            torch.cuda.empty_cache()

        S = S_sum / n_steps_accum

        figx = 7.5
        figy = 5.1
        ms = ms_sizes[i]
        fs_panel = 8.5
        cb_ticks_font_size = 5
        BZ_color = 'k'

        atoms = read('../data/c2db-4732.json')

        a_lat = np.linalg.norm(atoms.cell[0])
        c_lat = np.linalg.norm(atoms.cell[2])

        nk = cell
        a = a_lat*cell

        a_cell = elphmod.bravais.primitives(ibrav=4, angle=120, a = a_lat, c = c_lat)
        b1,b2,b3 = elphmod.bravais.reciprocals(*a_cell)

        gs_kw = dict(width_ratios=[1], height_ratios=[1])
        fig, (ax) = plt.subplots(1,1, figsize=(figx,figy), gridspec_kw=gs_kw)

        struc = dict()

        struc['class'] = S.reshape(cell*cell)

        outline = elphmod.bravais.BZ() 
        outline = np.array(outline) / (a / nk)
        outline = list(outline)

        #ax.fill_between(*zip(*outline), color = BZ_color)
        ax.plot(*zip(*outline), color='gray', linewidth = 1, alpha=0.4, zorder=1)

        S_plot = struc['class']

        S_q.append(S_plot[q_index])

        q_ = np.reshape(q,(cell*cell,3))
        qx = q_[:,0] / (2*np.pi)
        qy = q_[:,1] / (2*np.pi)
                
        if cell == 8:
            edge = [27,36,45,53,4,19,26,32,46]
            for shift_index in edge:
                qx = np.append(qx, np.array([qx[shift_index]]))
                qy = np.append(qy, np.array([-qy[shift_index]]))
                S_plot = np.append(S_plot, np.array([S_plot[shift_index]]))
                
        elif cell == 16:
            edge = [102,119,136,153,170,202,233,8,39,70,100,114,128,158,172]
            for shift_index in edge:
                qx = np.append(qx, np.array([qx[shift_index]]))
                qy = np.append(qy, np.array([-qy[shift_index]]))
                S_plot = np.append(S_plot, np.array([S_plot[shift_index]]))
                
        elif cell == 56:
            qx = np.append(qx, np.array([-qx[1482]]))
            qy = np.append(qy, np.array([qy[1482]]))
            S_plot = np.append(S_plot, np.array([S_plot[1482]]))

            qx = np.append(qx, np.array([-qx[1482]]))
            qy = np.append(qy, np.array([-qy[1482]]))
            S_plot = np.append(S_plot, np.array([S_plot[1482]]))

            edge = [1083,1140,1197,1254,1311,1368,1425,1482,1539,1596,1653,1710,1767,1824,1881,1938,1995,2052,2109,2165,2276,2387,2498,2609,2720,2831,2942,
                    3053,28,139,250,361,472,583,694,805,916,1027,1082,1136,1190,1244,1298,1352,1406,1460,1514,1568,1678,1786,1840,1894,1948,2002,2056,2110]
            for shift_index in edge:
                qx = np.append(qx, np.array([qx[shift_index]]))
                qy = np.append(qy, np.array([-qy[shift_index]]))
                S_plot = np.append(S_plot, np.array([S_plot[shift_index]]))
                
        elif cell == 64:
            edge = [1430,1495,1560,1625,1690,1755,1820,1885,1950,2015,2080,2145,2210,2275,2340,2405,2470,2535,2600,2665,2730,2858,2985,3112,3239,3366,3493,3620,3747,3874,4001,32,
                    159,286,413,540,667,794,921,1048,1175,1302,1428,1490,1552,1614,1676,1738,1800,1862,1924,1986,2048,2174,2236,2298,2360,2422,2484,2546,2608,2670,2732]
            for shift_index in edge:
                qx = np.append(qx, np.array([qx[shift_index]]))
                qy = np.append(qy, np.array([-qy[shift_index]]))
                S_plot = np.append(S_plot, np.array([S_plot[shift_index]]))

        S_all_q.append(np.array(S_plot))

        im = ax.scatter(qx, qy, c=abs(S_plot), s=ms, cmap='cubehelix_r', norm=colors.LogNorm(vmin=5e-8, vmax=1), zorder=2)

        cb = fig.colorbar(im, ax=ax, aspect=10)
        cb.ax.tick_params(labelsize=cb_ticks_font_size)

        ax.axis('off')

        plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)   
                    
        plt.tight_layout()
        outpath = './plots/both_{}K_{}x{}x1.png'.format(temp,cell,cell)
        plt.savefig(outpath)
        plt.close()
        
        print('{}x{}x1, {}K done!'.format(cell,cell,temp), flush=True)

    np.save('./data/both_S_q_{}x{}x1.npy'.format(cell,cell), np.array(S_q))
    np.save('./data/both_S_all_q_{}x{}x1.npy'.format(cell,cell), np.array(S_all_q))

    plt.scatter(temps, S_q)
    plt.xlabel("Temperature [K]")
    plt.ylabel(r"<$S(\mathbf{q} = (1/2, 0, 0), T)>$")
    plt.title("Structure Factor vs Temperature")
    plt.tight_layout()
    plt.savefig('./plots/both_q_{}x{}x1.png'.format(cell,cell))
    plt.close()
