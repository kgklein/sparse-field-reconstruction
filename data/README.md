Large simulation files are not stored in this repository.

Obtain the main simulation snapshot from the shared Box folder and place it in your
local data directory. This simulation input is used by both:

- reconstruction benchmarks that sample or reconstruct a 3D simulation field
- moving-observatory HelioSwarm time-series runs that sample the field along spacecraft trajectories

Simulation-backed HelioSwarm workflows also require local HelioSwarm CDF inputs in addition to
the simulation `.npy` snapshot.

For simulation-backed HelioSwarm runs:

- the simulation box extents are supplied at run time with CLI flags such as `--sim-box-x`, `--sim-box-y`, and `--sim-box-z` in units of `rho_p`
- HelioSwarm spacecraft positions originate in physical `km` and are transformed at run time using `--rho-p-km`

Then set:

export SPARSE_RECON_DATA=/absolute/path/to/data
