import numpy as np
import matplotlib.pyplot as plt
import ase.units
from ase.optimize import BFGS, FIRE
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from ase.io import write, read
import phonopy as ph
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from mace.calculators import MACECalculator
from phonopy.structure.cells import get_primitive, get_supercell
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms
from ase.build.supercells import make_supercell
from ase.build import sort
from copy import deepcopy

def get_crystal(structure):
    cell = PhonopyAtoms(symbols=structure.get_chemical_symbols(),
                       cell=structure.cell[:],
                       scaled_positions=structure.get_scaled_positions())
    return cell

def phonopy_pre_process(cell, supercell_matrix):
    smat = supercell_matrix
    phonon = ph.Phonopy(cell, smat,
                        primitive_matrix=[[1, 0.0, 0.0],
                                         [0.0, 1, 0.0],
                                         [0.0, 0.0, 1]],)
    phonon.generate_displacements(distance=0.015)
    return phonon

def run_calc(calc, phonon):
    supercells = phonon.supercells_with_displacements
    set_of_forces = []
    for scell in supercells:
        cell = Atoms(symbols=scell.get_chemical_symbols(),
                     scaled_positions=scell.get_scaled_positions(),
                     cell=scell.get_cell(),
                     pbc=True)
        cell.set_calculator(calc)
        forces = cell.get_forces()
        drift_force = forces.sum(axis=0)
        for force in forces:
            force -= drift_force / forces.shape[0]
        set_of_forces.append(forces)
    return set_of_forces

def phonopy_post_process(phonon, interval):
    phonon.set_mesh([20, 20, 1])
    phonon.run_total_dos(freq_min=-20,freq_max=115,freq_pitch=(115+20)/interval,use_tetrahedron_method=False,sigma=0.3)
    omege=[]
    dosevi=[]
    for omega, dos in np.array(phonon.get_total_DOS()).T:
        omege.append(omega)
        dosevi.append(dos)
    return omege,dosevi

E0_calc = MACECalculator(model_paths='../mace_model/TiSe2_model_6_run-123_swa.model', device='cpu')

atoms = read('../data/c2db-4732.json')
atoms = sort(atoms)
write('./POSCAR', atoms)

atoms.calc = E0_calc

dyn = BFGS(atoms)
dyn.run(fmax=1e-4)

write('./POSCAR_relaxed', atoms)

atoms.calc = None

cell = get_crystal(atoms)
phonon = phonopy_pre_process(cell, supercell_matrix=[[56,0,0], [0,56,0], [0,0,1]])
set_of_forces = run_calc(E0_calc, phonon)
phonon.produce_force_constants(forces=set_of_forces)
force_constants = None
force_constants = phonon.get_force_constants()

path = [[[0, 0, 0], [1/2, 0, 0], [1/3, 1/3, 0], [0, 0 ,0]]]
labels = ["$\\Gamma$", "M", "K", "$\\Gamma$"]
qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
phonon.run_band_structure(qpoints, path_connections=connections, labels=labels, with_eigenvectors=False, is_band_connection=False,)
phonon.run_mesh([20,20,1]) #[20,20,20]
phonon.run_total_dos()

phonon.plot_band_structure_and_dos().savefig('./harmonic_phonons.png')
phonon.plot_band_structure_and_dos().savefig('./harmonic_phonons.pdf')
plt.close()

write_FORCE_CONSTANTS(force_constants, './FORCE_CONSTANTS')
