from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.bussi import Bussi
from mace.calculators import MACECalculator
from ase.io import read
from ase.io.trajectory import Trajectory
import numpy as np

# Load MACE model calculator
E0_calc = MACECalculator(model_paths='../../mace_model/TiSe2_model_6_run-123_swa.model', default_dtype="float32", enable_cueq=True, device='cuda')

# Read atoms from file
atoms = read('../../data/SPOSCAR')
atoms.calc = E0_calc

# MD parameters
timestep = 1.5 * units.fs
taut = 50 * units.fs      # thermostat time constant
temperature = 25          # Kelvin

seed = 1234
num_steps = 110000

# Initialize velocities and remove net momentum once at start
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature, rng=np.random.RandomState(seed))
Stationary(atoms)

# Setup Bussi thermostat dynamics
dyn = Bussi(atoms, timestep, temperature_K=temperature, taut=taut, rng=np.random.RandomState(seed))

# Trajectory file
traj = Trajectory('./bussi.traj'.format(temperature), 'w', atoms)

# Combined function to print energies and remove net momentum periodically
def monitor_and_stationary(a=atoms, interval_stationary=10):
    epot = a.get_potential_energy()
    ekin = a.get_kinetic_energy()
    step = dyn.nsteps

    print(f'Step {step}: Epot = {epot:.10e} eV, Ekin = {ekin:.10e} eV, T = {ekin / (1.5 * len(a) * units.kB):.1f} K, Etot = {epot + ekin:.10e} eV', flush=True)
    traj.write(a)

    if step % interval_stationary == 0:
        Stationary(a)
        print(f"Removed net linear momentum at step: {step}", flush=True)

# Attach the combined function every step
dyn.attach(monitor_and_stationary, interval=1)

# Print initial energies
#monitor_and_stationary()

# Run the dynamics for num_steps steps
dyn.run(num_steps)
