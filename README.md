# OpenMM SAPT-ML Plugin

A plugin for the [OpenMM](https://openmm.org) molecular simulation toolkit that implements SAPT-ML, a flexible,
polarizable many-body water model: a SAPT-based force field corrected by two- and three-body permutationally
invariant polynomials fitted to CCSD(T) energies. It follows the layout of the OpenMM example plugin: one library
`SAPTMLPlugin` with the public API, Reference and CUDA platform implementations, C++ tests, SWIG-generated Python
wrappers, and a Python package (`saptml`) that registers the force-field XML tags, so a simulation is set up with the
standard `ForceField` / `createSystem` workflow.

The model is the sum of

| term | implementation | force group |
|---|---|---|
| one-body (Partridge-Schwenke surface, MB-pol scaling) | `PartridgeSchwenkeForce` (analytic) | 2 |
| electrostatics: geometry-dependent charges, Thole point dipoles, mutual induction, PME | `MBPolElectrostaticsForce` | 1 |
| SAPT-FF short range + Tang-Toennies damped C6-C12 dispersion | two `CustomNonbondedForce`s (`<SAPTFFForce>`) | 3 |
| two-body permutationally invariant polynomial | `WaterDimerPIPForce` | 6 |
| three-body permutationally invariant polynomial | `WaterTrimerPIPForce` | 5 |

## Installation

### Quick install (conda)

```bash
git clone https://github.com/hapu9734/openmm_saptml.git
cd openmm_saptml
bash install.sh                 # creates the conda environment "saptml", builds, installs and tests
conda activate saptml
```

`install.sh` creates a conda-forge environment with OpenMM (8.6.1 by default), the compilers, CMake, the CUDA headers and
the SWIG version that wrapped OpenMM, builds the plugin, runs the C++ tests from the build tree, installs the libraries
into the environment (`lib/`, `lib/plugins/`, `include/`) and the `saptml` package into its Python, and runs
`tests/test_saptml.py`. Run it on a machine with an NVIDIA GPU to test the CUDA platform. Options:

| option | meaning |
|---|---|
| `-n NAME` / `-P PATH` | environment name or path (created when missing, reused otherwise) |
| `-e` | build into the currently active environment as it is |
| `-o VERSION` | OpenMM version (default 8.6.1; 8.2 and later are supported) |
| `-p VERSION` | Python version for a new environment (default 3.11) |
| `-c VERSION` | CUDA version (default: the driver's, from `nvidia-smi`) |
| `-j N`, `-t`, `-k` | parallel build jobs, skip the tests, keep the build directory |
| `-m` | use `mamba` instead of `conda` |

The environment has about 100 packages (460 MB). With conda 23.10 or later (libmamba solver) it is created in a few
minutes; older conda versions use the classic solver, which can take an hour (update conda or use `-m`).

### Manual build

Requirements: OpenMM 8.2 or later, CMake 3.17, a C++11 compiler, SWIG (the version that wrapped your OpenMM: see the
header of `openmm/openmm.py`), and for the CUDA platform the CUDA toolkit headers and the OpenMM CUDA platform.

```bash
mkdir build && cd build
cmake .. -DOPENMM_DIR=$CONDA_PREFIX -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX
make -j8
make test                # TestReferenceSAPTMLForces, TestCudaSAPTMLForces{Single,Mixed,Double}
make install             # libSAPTMLPlugin, lib/plugins/libSAPTMLPlugin{Reference,CUDA}, headers
make PythonInstall       # pip-installs the saptml package with the SWIG module
```

Installed into the OpenMM prefix, the platform kernels are loaded by OpenMM at `import openmm`. For another prefix, set
`SAPTML_PLUGIN_DIR=<prefix>/lib/plugins`. `SAPTML_PATH` adds directories searched for the model files named in an XML.

## Running a simulation

```python
import saptml                                   # loads the kernels and registers the XML tags
import openmm as mm
from openmm import app, unit

pdb = app.PDBFile("input.pdb")
forcefield = app.ForceField(saptml.xml())      # the packaged model; or your own forcefield.xml
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.addExtraParticles(forcefield)         # adds the M sites
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9*unit.nanometer)
system.addForce(mm.MonteCarloBarostat(1*unit.bar, 298.15*unit.kelvin, 25))
integrator = mm.LangevinMiddleIntegrator(298.15*unit.kelvin, 1/unit.picosecond, 0.2*unit.femtosecond)
simulation = app.Simulation(modeller.topology, system, integrator, mm.Platform.getPlatformByName("CUDA"),
                            {"Precision": "mixed"})
simulation.context.setPositions(modeller.positions)
simulation.step(1000)
```

`examples/water256/run_md.py` is the minimal NVT script in the style of OpenMM's `simulatePdb.py` (2 ps at
298.15 K, trajectory in `output.dcd`). `examples/water256/run.py` is a complete NPT/NVT driver (Monte Carlo barostat,
isotropic or anisotropic; Langevin thermostat; box dipole log for the dielectric constant and IR spectra; restart from
its checkpoint):

```bash
cd examples/water256
python run_md.py                                  # 2 ps NVT at 298.15 K, 256 waters
python run.py --steps 50000 --outdir out          # 10 ps NPT at 298.15 K and 1 bar
python run.py -h
```

The water is flexible (no constraints, `rigidWater="false"` in the residue template), so the time step is 0.2 fs.
On an RTX 4080/5080 a 256-water box takes about 2.4 ms per step in mixed precision (about 7.4 ns/day).

### Induced-dipole solver

The attributes of `<MBPolElectrostaticsForce>` can be overridden by keyword arguments of `ForceField.createSystem`:

| attribute / keyword | default | meaning |
|---|---|---|
| `mutualInducedMaxIterations` | 60 | maximum number of induction iterations |
| `mutualInducedTargetEpsilon` | 1e-5 | convergence target: RMS change of the induced dipoles between iterations (Debye) |
| `inducedDipoleMethod` | `DIIS` | `DIIS` (direct inversion in the iterative subspace) or `SOR` (successive over-relaxation) |
| `diisHistorySize` | 20 | number of DIIS vectors (2-32) |
| `sorParameter` | 0.55 | over-relaxation parameter (1 = Jacobi) |
| `predictorOrder` | 3 | order of the initial-guess extrapolation from previous steps (0 = start from the direct dipoles) |
| `ewaldErrorTolerance` | 1e-5 | PME accuracy |

The defaults are meant for a Langevin thermostat with a friction of about 1/ps. With an extrapolated initial guess and
a loose target the forces are not exactly conservative: in NVE a 256-water box at 0.2 fs gains about 4 kJ/mol/ps with the
defaults, which a weak thermostat (0.1/ps) turns into a temperature offset of about 2 K. For NVE or weakly coupled runs
(dynamics, spectra) use `mutualInducedTargetEpsilon=1e-6`, preferably with `predictorOrder=2`: the drift vanishes
(below 0.02 kJ/mol/ps) at about 10 % extra cost.

### Force-field XML

`saptml/data/saptml.xml` (written by `tools/write_xml.py`) defines the atom types, the residue template with the
`average3` virtual site, and the five force tags. `<PartridgeSchwenkeForce>` creates the analytic one-body force.
The PIP tags take `model="file"` (`.wdpip` / `.wtpip`, format documented in `saptml/pipmodel.py`), `maxNeighbors`,
`overflowCheckInterval`, and the switching ranges `oxygenOxygenSwitch`, `intermolecularOHSwitch`,
`intermolecularHHSwitch`, `intramolecularOHWindow`, `intramolecularHHWindow` (nm).

### Python API

`saptml.MBPolElectrostaticsForce`, `saptml.PartridgeSchwenkeForce`, `saptml.WaterDimerPIPForce` and
`saptml.WaterTrimerPIPForce` are the wrapped classes (`cast` / `isinstance` as in the OpenMM plugins);
`saptml.forcefield.forces(system)` returns the SAPT-ML forces of a System by name. `force.getInducedDipoles(context)` and
`force.getCharges(context)` return the induced dipoles (e nm) and the geometry-dependent charges (e) of the last
evaluation; `saptml.total_dipole(context)` is the box dipole for dielectric constants and IR spectra.

## Tests

- `make test` (C++): energies against stored references, forces against finite differences, CUDA against Reference,
  solver variants, a few dynamics steps.
- `python tests/test_saptml.py`: the installed package against reference energies of an independent implementation
  (12-water cluster on the Reference platform; the 256-water example box on CUDA in double precision) and 1000 steps of
  NPT dynamics.

## Layout

```
openmmapi/          public API (include/) and the Force / ForceImpl classes (src/)
platforms/reference CPU implementation and tests
platforms/common    host code shared by the GPU platforms, device kernels (src/kernels/*.cc)
platforms/cuda      CUDA kernel factory and tests
python/             SWIG interface (saptmlplugin.i), setup.py, the saptml package and its data files
tests/              shared C++ test cases, reference geometries and energies, test_saptml.py
examples/water256/  run_md.py (minimal NVT), run.py (full driver) and a 256-water input.pdb
tools/write_xml.py  writes a force-field XML from a parameter set
install.sh          conda build and install script
```

## Citation

If you use this plugin, please cite

- S. Bae and C. Y. Son, "Δ-ML Corrected Many-Body Water Model from Symmetry-Adapted Perturbation Theory"
  (in preparation);
- P. Eastman et al., "OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials",
  J. Phys. Chem. B 128, 109 (2024).

## Authors

- Sangwon Bae ([ORCID 0009-0009-7169-8091](https://orcid.org/0009-0009-7169-8091)), sangone@snu.ac.kr
- Chang Yun Son ([ORCID 0000-0002-0784-6565](https://orcid.org/0000-0002-0784-6565)), changyunson@snu.ac.kr
  (corresponding author)

Department of Chemistry, Seoul National University, Seoul 08826, Republic of Korea

## License

MIT License, see `LICENSE`. Code derived from OpenMM, the MB-pol OpenMM plugin and FFTPACK keeps its original
notices, listed at the end of `LICENSE`.
