import numpy as np
from ase.io import read

# Load training data
frames = read("./TiSe2_frames_train_MACE.xyz", index=":")

A = []
b = []

for atoms in frames:
    symbols = atoms.get_chemical_symbols()
    n_ti = symbols.count("Ti")
    n_se = symbols.count("Se")
    
    A.append([n_ti, n_se])
    b.append(atoms.get_potential_energy()) #info["energy"])

A = np.array(A)
b = np.array(b)

# Solve A * x = b via least-squares regression
E0_ti, E0_se = np.linalg.lstsq(A, b, rcond=None)[0]

print(f"Calculated r2SCAN E0 values:")
print(f'--E0s=\'{{"Ti": {E0_ti:.6f}, "Se": {E0_se:.6f}}}\'')
