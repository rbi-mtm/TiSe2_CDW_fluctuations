import cellconstructor as CC
import cellconstructor.Phonons
import numpy as np
from ase.io import read, write
import matplotlib.pyplot as plt

temps = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 370, 380, 390, 400, 420, 440, 460, 480, 500]

reference_ti = [[-1.7681019000598672,3.0624423238627592,9.0525905049867710], 
                [1.7681019000598672,3.0624423238627592,9.0525905049867710],
                [3.5362038001197345,0.0,9.0525905049867710], 
                [0.0,0.0,9.0525905049867710]]

distances, average_distances = [], []

for temp in temps:
    struct = CC.Phonons.Phonons('./final_dyn_{}K_'.format(temp), 4)

    ti_indices = [i for i, atom in enumerate(struct.structure.atoms) if atom == 'Ti']

    hlp = []

    struct_ti = struct.structure.coords[ti_indices]

    for i in range(len(struct_ti)):
        hlp.append(np.linalg.norm(struct_ti[i][:2] - reference_ti[i][:2]))

    distances.append(hlp)
    average_distances.append(np.mean(np.array(hlp)[:3]))

print('\n')
print(temps)
print('\n')
for dst in distances:
    print(dst)
print('\n')
for avg_dst in average_distances:
    print(avg_dst)

np.save('./ti_average_distances_sscha.npy', average_distances)

plt.scatter(temps, average_distances)
plt.xlabel('Temperature (K)')
plt.ylabel(r'$\delta$ Ti (Å)')
plt.savefig('./ti_average_displacement.png')
plt.close()
