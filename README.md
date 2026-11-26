# Fluctuation-driven multi-step charge density wave melting in monolayer TiSe<sub>2</sub>

Data and code for reproducing the results of the study of charge density wave (CDW) fluctuations in monolayer TiSe<sub>2</sub>:

> L. Benić, D. Novko, I. Lončarić, *Fluctuation-driven multi-step charge density wave melting in monolayer TiSe<sub>2</sub>*, [arXiv:2604.20355](https://arxiv.org/abs/2604.20355).

The same material is archived on Zenodo: [10.5281/zenodo.24517322](https://doi.org/10.5281/zenodo.24517322).

## Installation

`conda_env.yaml` is the Conda environment with all packages needed to run the codes:

```bash
conda env create -f conda_env.yaml
conda activate conda_env
```

## Contents

| Path | Description |
|---|---|
| `conda_env.yaml` | Conda environment |
| `data/` | Structures used by the analysis codes (see below) |
| `bussi_md/` | Code for reproducing the MD trajectories (example for 25 K) |
| `mace_model/` | Training script, training and test datasets, and the trained MACE potential |
| `mace_model_r2scan/` | Training script, training and test datasets, and the trained MACE potential obtained by fine-tuning the original MACE potential using r<sup>2</sup>SCAN data; `TiSe2_model_6.model` is the PBE-based MACE potential used as the starting (foundation) model for the fine-tuning |
| `dft_dataset/` | Input files for the VASP calculations used for the construction of the dataset |
| `structure_factors/` | Structure factors from MD data |
| `dynaphopy/` | Phonon dispersions and power spectra from MD data (DynaPhoPy) |
| `rdf/` | Radial distribution functions |
| `sscha/` | SSCHA calculations |
| `*.py` in the root | Displacements, folded positions and CDW order parameter (see below) |

### Structures (`data/`)

| File | Description |
|---|---|
| `c2db-4732.json` | Unit cell of the normal phase structure of monolayer TiSe<sub>2</sub> |
| `POSCAR_cdw` | 2x2x1 CDW structure |
| `POSCAR_avg_25K` | Example of an averaged structure obtained from the MD data at 25 K |
| `SPOSCAR` | 56x56x1 supercell of the normal phase structure in direct coordinates, used for starting the MD simulations |
| `SPOSCAR_cartesian` | The same 56x56x1 supercell in Cartesian coordinates |
| `SPOSCAR_cdw` | 28x28x1 supercell of the CDW phase structure in direct coordinates |
| `eigvec_Q0.npz`, `eigvec_Q1.npz`, `eigvec_Q2.npz` | Eigenvectors of the soft phonon mode at the three M points, used by `phase_gaussian_sigma4.py` and `phase_global.py` |

### Codes

| File | Description |
|---|---|
| `average_displacements.py` | Average atomic displacements |
| `distances_average_positions.py` | Distances between average atomic positions and normal phase atoms |
| `positions_folded.py` | Atomic positions folded onto a supercell of arbitrary size |
| `phase_gaussian_sigma4.py` | Amplitudes and phases of the CDW order parameters with a Gaussian of width sigma = 4 Å, from MD data |
| `phase_global.py` | Amplitudes and phases of the CDW order parameters without the Gaussian, from MD data |
| `structure_factors/structure_factors.py` | Structure factors from MD data |
| `dynaphopy/dynaphopy_analysis.py` | DynaPhoPy analysis |
| `dynaphopy/harmonic.py` | Harmonic force constants |
| `dynaphopy/phonon_dispersions.py` | Phonon dispersions from MD data |
| `dynaphopy/xdatcar_from_traj.py` | Conversion of `.traj` into XDATCAR files |
| `rdf/rdf_Ti.py` | Radial distribution functions |
| `rdf/gaussian_fitting.py` | Fitting of Gaussians to the radial distribution functions |
| `sscha/TiSe2_SSCHA.py` | SSCHA calculations |
| `sscha/ti_displacements.py` | Atomic displacements from SSCHA data |

## Usage

All paths in the codes are relative, so the codes in the root are run from the root of the repository, and the codes in the subdirectories from their own directory.

1. **MD trajectories.** Due to their large size, the MD trajectories are not provided, only the code for reproducing them. `bussi_md/25/bussi.py` is the example for 25 K; for other temperatures copy it to `bussi_md/<T>/` and change the temperature. The analysis codes expect the trajectories in `bussi_md/<T>/bussi.traj`.
2. **Analysis.** For example:

   ```bash
   python average_displacements.py
   python phase_gaussian_sigma4.py            # all temperatures
   python phase_gaussian_sigma4.py 100 200    # selected temperatures
   python phase_global.py
   ```

   `phase_gaussian_sigma4.py` and `phase_global.py` read the eigenvectors of the soft phonon mode at the three M points from `data/eigvec_Q0.npz`, `data/eigvec_Q1.npz` and `data/eigvec_Q2.npz`, and write the results to `order_parameter_gaussian_sigma4/` and `order_parameter_global/`.

## Notes

- **VASP inputs.** The `KPOINTS` and `POSCAR` files in `dft_dataset/` are given for the unit cell structure, and they are changed according to the supercell that is used. The `POTCAR` file is not included, since VASP pseudopotentials cannot be redistributed; it has to be generated from your own VASP installation using the PAW_PBE Se (06Sep2000) and PAW_PBE Ti (08Apr2002) potentials, in the order Se, Ti as in the `POSCAR` file.
- **Training scripts.** The `PYPATH` variable in the training scripts in `mace_model/` and `mace_model_r2scan/` has to be changed to the path of your own MACE installation.

## Citation

If you use this data or code, please cite the paper above and the Zenodo record [10.5281/zenodo.24517322](https://doi.org/10.5281/zenodo.24517322).
