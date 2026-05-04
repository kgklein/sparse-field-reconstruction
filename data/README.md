Large simulation files are not stored in this repository.

Obtain the main simulation snapshot from the shared Box folder and place it in your
local data directory. This simulation input is used by both:

- reconstruction benchmarks that sample or reconstruct a 3D simulation field
- moving-observatory HelioSwarm time-series runs that sample the field along spacecraft trajectories

Different workflows need different local assets.

## Asset Types

### Magnetic-field simulation snapshots

These are the standard field inputs used by:

- simulation-backed reconstruction benchmarks
- magnetic-field moving-observatory time-series runs
- simulation-cube structure-function runs

Supported examples include:

- `.npy` snapshots containing a structured vector field
- `.bp` field snapshots when the required vector variable names are supplied at run time

### Velocity, density, and background-field inputs

Velocity/Elsasser-capable time-series workflows may require additional local inputs:

- a field snapshot for magnetic geometry or secondary magnetic sampling
- an ion-moments snapshot for packed variables such as `nux,nuy,nuz` and scalar density `n`
- an optional Lua file describing the background magnetic field when `.bp` field inputs require it

These assets are used by workflows that want sampled velocity time series plus derived Elsasser pair products for lag-tetrahedra analysis.

### HelioSwarm CDF inputs

Simulation-backed HelioSwarm workflows also require local HelioSwarm CDF inputs in addition to
the simulation data products above.

HelioSwarm inputs can be:

- a single `.cdf` file
- a directory containing monthly `.cdf` summary files

They are required by:

- static HelioSwarm sampling in reconstruction benchmarks
- moving-observatory time-series runs
- any downstream analysis built from those time-series outputs

For simulation-backed HelioSwarm runs:

- the simulation box extents are supplied at run time with CLI flags such as `--sim-box-x`, `--sim-box-y`, and `--sim-box-z` in units of `rho_p`
- HelioSwarm spacecraft positions originate in physical `km` and are transformed at run time using `--rho-p-km`

## Workflow To Asset Map

- Synthetic reconstruction benchmark:
  - no local data files required
- Simulation-backed reconstruction benchmark:
  - magnetic-field simulation snapshot
- HelioSwarm-backed reconstruction benchmark on a synthetic field:
  - HelioSwarm CDF input
- HelioSwarm-backed reconstruction benchmark on a simulation field:
  - magnetic-field simulation snapshot
  - HelioSwarm CDF input
- Magnetic-field moving-observatory time-series run:
  - magnetic-field simulation snapshot
  - HelioSwarm CDF input
- Velocity/Elsasser moving-observatory time-series run:
  - field snapshot
  - ion-moments snapshot
  - HelioSwarm CDF input
  - optional background-field Lua file
- Structure functions from an existing time-series CSV:
  - previously generated `helioswarm_timeseries.csv`
  - optional `helioswarm_timeseries_metadata.json`
- Space-time correlation:
  - previously generated `helioswarm_timeseries.csv`
  - optional `helioswarm_timeseries_metadata.json`
- Lag tetrahedra:
  - previously generated `helioswarm_timeseries_metadata.json`
  - previously generated `helioswarm_timeseries_elsasser_pairs.npz`
  - optional `helioswarm_timeseries_elsasser_pairs.json`

## Local Environment

Then set:

```bash
export SPARSE_RECON_DATA=/absolute/path/to/data
```
