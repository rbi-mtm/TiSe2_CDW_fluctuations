import freud
import numpy as np
from ase.io import Trajectory
from ase.io import read
import matplotlib.pyplot as plt

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

starts = [10000] * len(temps)
ends = [110000] * len(temps)

for i, temp in enumerate(temps):
    print('{}K'.format(temp), flush=True)

    traj = read('../bussi_md/{}/bussi.traj'.format(temp), index=(slice(starts[i],ends[i])))

    print('reading done!', flush=True)

    rdf = freud.density.RDF(bins=2000, r_max=5) #8)

    for atoms in traj:
        cell = atoms.cell[:].astype(np.float32)
        box = freud.box.Box.from_matrix(cell)

        symbols = atoms.get_chemical_symbols()
        ti_positions = atoms.positions[np.array(symbols) == "Ti"].astype(np.float32)

        rdf.compute((box, ti_positions))

    print('loop done!\n', flush=True)

    r = rdf.bin_centers
    g_r = rdf.rdf

    np.save('./data/dists_Ti_{}K.npy'.format(temp), np.array(r))
    np.save('./data/rdf_Ti_{}K.npy'.format(temp), np.array(g_r))

    plt.plot(r, g_r)
    plt.xlabel('r')
    plt.ylabel('g_r')
    plt.xlim(3, 4)
    plt.savefig('./plots/rdf_Ti_{}K.png'.format(temp))
    plt.close()
