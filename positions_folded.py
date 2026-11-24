from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from ase.build import make_supercell
import argparse

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

starts = [10000] * len(temps)
ends = [110000] * len(temps)

atoms_unit = read('./data/c2db-4732.json')
unit_cell = atoms_unit.get_cell()

lx, ly, lz = unit_cell.lengths()

cell = 4

M = [[cell,0,0],[0,cell,0],[0,0,1]]

supercell = make_supercell(atoms_unit, M)
supercell_cell = supercell.get_cell()

symbols_supercell = supercell.get_chemical_symbols()
ti_indices_supercell = [i for i, s in enumerate(symbols_supercell) if s=='Ti']

R_ti_supercell = supercell.get_positions(wrap=False)[ti_indices_supercell]

R_ti_supercell[:,0] += lx/2 - ly*np.sqrt(3)/4
R_ti_supercell[:,1] += ly/2

def wrap_to(trajfile, path, cell, ind, indices=':'):
    traj = read(trajfile, index=indices)
    
    for atom in traj:
        atom.positions[:,0] += lx/2 - ly*np.sqrt(3)/4 
        atom.positions[:,1] += ly/2 
        atom.cell = supercell_cell
        atom.wrap()

    write(path+'bussi_wrapped_{}x{}x1_cell_{}K.traj'.format(cell,cell,ind), traj)

def extract_positions(trajfile, indices=':'):
    R_ti = []
    traj = read(trajfile, index=indices)

    for atoms in traj:
        symbols = atoms.get_chemical_symbols()
        ti_indices = [i for i, s in enumerate(symbols) if s=='Ti']

        R = atoms.get_positions(wrap=False)[ti_indices]
        R_ti.append(R)

    return np.array(R_ti)

xmin, xmax, ymin, ymax = np.inf, -np.inf, np.inf, -np.inf

for i, temp in enumerate(temps):
    wrap_to(trajfile='./bussi_md/{}/bussi.traj'.format(temp), path='./bussi_md/{}/'.format(temp), cell, ind=temp, indices=(slice(starts[i],ends[i]))) 

    positions = extract_positions(trajfile='./bussi_md/{}/bussi_wrapped_{}x{}x1_cell_{}K.traj'.format(temp,cell,cell,temp), indices=':')
    np.save('./bussi_md/{}/positions_{}x{}x1_cell_{}K.npy'.format(temp,cell,cell,temp), positions)

    x = positions[:, :, 0].ravel()
    y = positions[:, :, 1].ravel()

    xmin, xmax = min(xmin, np.min(x)), max(xmax, np.max(x))
    ymin, ymax = min(ymin, np.min(y)), max(ymax, np.max(y))

print('xmin = ', xmin, flush=True)
print('xmax = ', xmax, flush=True)
print('ymin = ', ymin, flush=True)
print('ymax = ', ymax, flush=True)

x_range = xmax - xmin
y_range = ymax - ymin

aspect = x_range / y_range

for i, temp in enumerate(temps):
    positions = np.load('./bussi_md/{}/positions_{}x{}x1_cell_{}K.npy'.format(temp,cell,cell,temp))

    x = positions[:, :, 0].ravel()
    y = positions[:, :, 1].ravel()

    base_height = 8
    fig, ax = plt.subplots(figsize=(base_height * aspect, base_height))
    counts, xedges, yedges, im = ax.hist2d(x, y, bins=5000, range=[[xmin, xmax], [ymin, ymax]], cmap='gist_heat')
    ax.scatter(R_ti_supercell[:,0], R_ti_supercell[:,1], color='green', s=25)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x position (Å)')
    ax.set_ylabel('y position (Å)')
    ax.set_title('2D Histogram of atomic positions in {}x{}x1 supercell at {} K'.format(cell,cell,temp))
    plt.tight_layout()
    plt.savefig('./ti_positions_{}x{}x1_cell_{}K.png'.format(cell,cell,temp))
    plt.close()
