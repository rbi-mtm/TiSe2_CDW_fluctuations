import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize

k_b = 1.38064852e-23

temps = [25, 50, 75, 100, 125, 150, 175, 185, 195, 200, 205, 210, 215, 225, 235, 250, 275, 300, 325, 350, 375, 400, 425, 450]

factors = np.linspace(1, len(temps), len(temps))[::-1]

starts = [10000] * len(temps)
ends = [110000] * len(temps)

delta_freqs = [(2*np.pi*1000)/(1.5*100000)] * len(temps)

delta_freqs_ = ['0.042'] * len(temps)

coeffs = [1000] * len(temps)

meshes = ['20x20x1'] * len(temps)

global_min, global_max = np.inf, -np.inf

for i in range(len(temps)):
    arr = np.load('./power_spectra_qs/spectrum_{}K_{}_{}_{}.npy'.format(temps[i], starts[i], ends[i], delta_freqs_[i]))
    arr = arr / temps[i]
    local_min = arr.min()
    local_max = arr.max()
    global_min = min(global_min, local_min)
    global_max = max(global_max, local_max)

print("global min:", global_min, flush=True)
print("global max:", global_max, flush=True)

for i in range(len(temps)):
    arr = np.load('./power_spectra_qs/spectrum_{}K_{}_{}_{}.npy'.format(temps[i], starts[i], ends[i], delta_freqs_[i]))

    data = arr.T

    data_norm = data / temps[i]

    x_axis = np.arange(data.shape[1])
    y_axis = np.load('./power_spectra_qs/freq_{}K_{}_{}_{}.npy'.format(temps[i], starts[i], ends[i], delta_freqs_[i]))

    x_tick_positions = [0, 50, 100, 150]
    x_tick_labels = ['$\\Gamma$', 'M', 'K', '$\\Gamma$']

    plt.imshow(data_norm, aspect='auto', origin='lower', extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]], cmap='cubehelix_r', interpolation='none', vmin=0, vmax=3.5e-4)
    plt.colorbar(label='Intensity [arb. units]')
    plt.xticks(x_tick_positions, x_tick_labels)
    for ind in range(1, len(x_tick_positions)-1):
        plt.axvline(x=x_tick_positions[ind], color='black', linestyle='--', linewidth=0.5) 
    plt.ylabel(r'$\omega$ [THz]')
    plt.ylim(min(y_axis), 10)
    plt.savefig('./plots/phonon_dispersions_{}K_{}_{}_{}.png'.format(temps[i], starts[i], ends[i], delta_freqs_[i]))
    plt.close()
