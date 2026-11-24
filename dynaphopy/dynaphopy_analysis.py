import os
import numpy as np
import phonopy.file_IO as file_IO
import dynaphopy.interface.iofile as io
import dynaphopy.interface.iofile.trajectory_parsers as parsers
import dynaphopy
from dynaphopy.interface.phonopy_link import get_force_sets_from_file, get_force_constants_from_file
from phonopy.file_IO import parse_FORCE_CONSTANTS, write_FORCE_CONSTANTS
from dynaphopy.interface.phonopy_link import ForceConstants
import matplotlib.pyplot as plt
from dynaphopy.dynamics import Dynamics
from ase.io import read, write
import ase.units as units
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--temperature', type=int, nargs="+", required=True)
args = parser.parse_args()

def fake_show(filepath):
    def inner():
        plt.savefig(f"{filepath}.png")
        plt.close()
    return inner

qs = np.load('../structure_factors/k_and_q/qs_51.npy')

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

starts = [10000] * len(temps)
ends = [110000] * len(temps)

delta_freqs = [(2*np.pi*1000)/(1.5*100000)] * len(temps)

delta_freqs_ = ['0.042'] * len(temps)

coeffs = [1000] * len(temps)

meshes = [(20,20,1)] * len(temps)

meshes_ = ['20x20x1'] * len(temps)

for i, temp in enumerate(temps):
    print('temperature = {}K, start = {}, end = {}, delta_freq = {}'.format(temps[i], starts[i], ends[i], delta_freqs_[i]), flush=True)

    structure = io.read_from_file_structure_poscar('./POSCAR_relaxed')

    '''structure.set_primitive_matrix([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])'''

    force_constants_phonopy = parse_FORCE_CONSTANTS(filename='./FORCE_CONSTANTS')
    force_constants = ForceConstants(force_constants = force_constants_phonopy, supercell = [[56, 0, 0], [0, 56, 0], [0, 0, 1]])

    structure.set_force_constants(force_constants)

    trajectory = parsers.read_VASP_XDATCAR('../bussi_md/{}/XDATCAR'.format(temp), structure, initial_cut=starts[i], end_cut=ends[i], time_step=0.0015)
    trajectory_ = read('../bussi_md/{}/bussi.traj'.format(temp), index='{}:{}'.format(starts[i]-1, ends[i]))

    trajectory.velocity = None

    vels = [] 
    for j in range(len(trajectory_)):
        vels.append(trajectory_[j].get_velocities())
    vels = np.array(vels)

    trajectory.velocity = vels*units.fs*1000

    quasiparticle = dynaphopy.Quasiparticle(trajectory)

    quasiparticle.parameters.mesh_phonopy = meshes[i]
    quasiparticle.parameters.silent = False
    quasiparticle.fitting_function = 0
    quasiparticle.number_of_bins_histogram = 100
    quasiparticle.band_resolution = 100
    quasiparticle.set_temperature(temps[i])

    quasiparticle.select_power_spectra_algorithm(3)
    quasiparticle.set_frequency_limits([0,10])
    quasiparticle.set_number_of_mem_coefficients(coeffs[i])
    quasiparticle.set_spectra_resolution(delta_freqs[i])
    
    quasiparticle.set_band_ranges({'ranges': [[[0.0, 0.0, 0.0], [1/2, 0.0, 0.0]], [[1/2, 0.0, 0.0], [1/3, 1/3, 0.0]], [[1/3, 1/3, 0.0], [0.0, 0.0, 0.0]]],
                                   'labels': [['$\\Gamma$', 'M'],['M', 'K'],['K', '$\\Gamma$']]})

    pswv = []

    for ind in range(len(qs)):

        print(quasiparticle.parameters._spectrum_resolution, flush=True)
        print(quasiparticle.parameters._frequency_range, flush=True)

        quasiparticle.set_reduced_q_vector(tuple(qs[ind]))

        pswv.append(quasiparticle.get_power_spectrum_wave_vector())

        if ind == 0:
            np.save(os.path.join('./power_spectra_qs', 'freq_{}K_{}_{}_{}'.format(temps[i], starts[i], ends[i], delta_freqs_[i])), quasiparticle.get_frequency_range())

        plt.show = fake_show(filepath = os.path.join('./plots', 'spectrum_wave_vector_{}K_{}_{}_{}_{}'.format(temps[i], starts[i], ends[i], delta_freqs_[i], ind)))
        
        quasiparticle.plot_power_spectrum_wave_vector()

    np.save(os.path.join('./power_spectra_qs', 'spectrum_{}K_{}_{}_{}'.format(temps[i], starts[i], ends[i], delta_freqs_[i])), np.array(pswv))
