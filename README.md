# Sparse Field Reconstruction

A benchmarking and analysis toolkit for sparse vector-field reconstruction, HelioSwarm sampling, and downstream turbulence diagnostics.

This repository started as a reconstruction benchmark framework and now supports a broader workflow:

- sparse reconstruction benchmarks on synthetic or simulation-backed fields
- HelioSwarm-driven sampling geometries for static reconstructions
- moving-observatory HelioSwarm time-series generation through a static simulation box
- downstream structure-function, space-time decorrelation, and lag-tetrahedra analyses

The main onboarding doc is this README. For a more operational view of the run modes, inputs, outputs, and workflow chaining, see [docs/workflows.md](/home/kgklein/Codes/sparse-field-reconstruction/docs/workflows.md).

## Setup

For a simple local install in this workspace:

```bash
cd /home/kgklein/Codes/sparse-field-reconstruction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For full development and notebook extras:

```bash
git clone https://github.com/kgklein/sparse-field-reconstruction.git
cd sparse-field-reconstruction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebooks]"
```

Most example runs in this repo use:

- `PYTHONPATH=src` so the package can be imported directly from the checkout
- `MPLCONFIGDIR=/tmp/mpl` so Matplotlib has a writable config/cache directory

## Workflow Families

The current codebase supports four main workflow families:

1. Reconstruction benchmarks via [`scripts/run_baseline.py`](/home/kgklein/Codes/sparse-field-reconstruction/scripts/run_baseline.py)
2. Moving-observatory HelioSwarm time-series generation via [`scripts/run_hs_timeseries.py`](/home/kgklein/Codes/sparse-field-reconstruction/scripts/run_hs_timeseries.py)
3. Structure-function analysis via [`scripts/run_structure_functions.py`](/home/kgklein/Codes/sparse-field-reconstruction/scripts/run_structure_functions.py)
4. Space-time decorrelation and lag-tetrahedra analysis via [`scripts/run_space_time_correlation.py`](/home/kgklein/Codes/sparse-field-reconstruction/scripts/run_space_time_correlation.py) and [`scripts/run_lag_tetrahedra.py`](/home/kgklein/Codes/sparse-field-reconstruction/scripts/run_lag_tetrahedra.py)

## Which Script Should I Run?

- Use `scripts/run_baseline.py` if you want to compare reconstruction methods on synthetic fields or a static simulation snapshot.
- Use `scripts/run_baseline.py` with `--hs-path` and `--hs-time` if you want a static HelioSwarm sampling geometry for a reconstruction benchmark.
- Use `scripts/run_hs_timeseries.py` if you want a 9-spacecraft moving-observatory time series through a simulation box.
- Use `scripts/run_structure_functions.py --input-mode interpolated_timeseries` if you want structure functions from an existing HelioSwarm time-series CSV.
- Use `scripts/run_structure_functions.py --input-mode simulation_cube` if you want structure functions sampled directly from a simulation cube.
- Use `scripts/run_space_time_correlation.py` if you want a multipoint decorrelation map from an existing HelioSwarm time series.
- Use `scripts/run_lag_tetrahedra.py` if you want lag-tetrahedra and Yaglom-style diagnostics from saved Elsasser pair products.

## Quick Runs

The repo includes small worked examples as shell scripts:

- [`quick_run.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run.sh): 2D synthetic smoke run
- [`quick_run_3d.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_3d.sh): 3D synthetic smoke run
- [`quick_run_3d_hs.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_3d_hs.sh): 3D synthetic field with HelioSwarm sampling
- [`quick_run_3d_sim_hs.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_3d_sim_hs.sh): simulation-backed reconstruction with HelioSwarm sampling
- [`quick_run_hs_timeseries.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_hs_timeseries.sh): magnetic-field time-series sampling run
- [`quick_run_hs_timeseries_velocity.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_hs_timeseries_velocity.sh): velocity-focused time-series run that also writes Elsasser pair products
- [`quick_run_structure_functions.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_structure_functions.sh): structure functions from an existing time-series product
- [`quick_run_structure_functions_cube.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_structure_functions_cube.sh): structure functions sampled directly from a simulation cube
- [`quick_run_space_time_correlation.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_space_time_correlation.sh): multipoint space-time decorrelation analysis
- [`quick_run_lag_tetrahedra.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_lag_tetrahedra.sh): lag-tetrahedra analysis from saved Elsasser pair products

You can run them directly, for example:

```bash
./quick_run.sh
./quick_run_3d.sh
./quick_run_hs_timeseries.sh
./quick_run_hs_timeseries_velocity.sh
./quick_run_structure_functions.sh
./quick_run_space_time_correlation.sh
./quick_run_lag_tetrahedra.sh
```

## Example Commands

2D synthetic smoke run:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --methods rbf \
  --sample-counts 24 \
  --geometries random \
  --noise-levels 0.0 \
  --nx 24 \
  --ny 24 \
  --output-dir /tmp/sparse_recon_smoke
```

3D synthetic smoke run:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --field-kind smooth_3d \
  --methods rbf \
  --sample-counts 48 \
  --geometries random,multi_probe_like \
  --noise-levels 0.0 \
  --nx 10 \
  --ny 9 \
  --nz 8 \
  --output-dir /tmp/sparse_recon_3d_smoke
```

3D synthetic field with HelioSwarm sampling:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --field-kind smooth_3d \
  --methods rbf \
  --noise-levels 0.0 \
  --nx 10 \
  --ny 9 \
  --nz 8 \
  --hs-path /path/to/HelioSwarm/HS-RT/PhB_SRD5B_0x75b \
  --hs-time "2029-06-26 00:00:00" \
  --output-dir /tmp/sparse_recon_hs_smoke
```

3D simulation snapshot with HelioSwarm sampling:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_baseline.py \
  --data-source simulation \
  --simulation-path /path/to/ot3D_field_75.npy \
  --methods rbf \
  --noise-levels 0.0 \
  --hs-path /path/to/HelioSwarm/DRM \
  --hs-time "2029-09-26 00:00:00" \
  --rho-p-km 100 \
  --sim-box-x 314.15926 \
  --sim-box-y 314.15926 \
  --sim-box-z 1570.79632 \
  --include-hub \
  --output-dir /tmp/sparse_recon_sim_hs
```

Moving-observatory magnetic-field time-series run:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_hs_timeseries.py \
  --simulation-path /path/to/ot3D_field_75.npy \
  --hs-path /path/to/HelioSwarm/DRM \
  --hs-time "2029-09-26 00:00:00" \
  --rho-p-km 100 \
  --sim-box-x 314.15926 \
  --sim-box-y 314.15926 \
  --sim-box-z 1570.79632 \
  --vx-kms 250 \
  --vy-kms 120 \
  --vz-kms 150 \
  --dt-seconds 0.03125 \
  --n-steps 10000 \
  --sampling-method trilinear \
  --plot-timeseries \
  --output-dir figs
```

Velocity/Elsasser time-series run:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_hs_timeseries.py \
  --simulation-path /path/to/ot3D_field_75.bp \
  --ion-moments-path /path/to/ot3D_ion_75.bp \
  --background-b-lua-path /path/to/ot3D.lua \
  --simulation-vector-vars nux,nuy,nuz \
  --simulation-density-var n \
  --geometry-vector-vars bx,by,bz \
  --secondary-timeseries-vector-vars bx,by,bz \
  --secondary-timeseries-component-labels "B_x,B_y,B_z" \
  --simulation-component-labels "u_x,u_y,u_z" \
  --hs-path /path/to/HelioSwarm/DRM \
  --hs-time "2029-09-26 00:00:00" \
  --rho-p-km 100 \
  --sim-box-x 314.15926 \
  --sim-box-y 314.15926 \
  --sim-box-z 1570.79632 \
  --vx-kms 250 \
  --vy-kms 120 \
  --vz-kms 150 \
  --dt-seconds 0.03125 \
  --n-steps 10000 \
  --sampling-method trilinear \
  --plot-timeseries \
  --output-dir figs_velocity
```

This velocity-focused mode can write both the sampled primary time series and the saved Elsasser pair products needed by the lag-tetrahedra workflow.

Structure-function analysis from an existing moving-observatory time-series product:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_structure_functions.py \
  --input-mode interpolated_timeseries \
  --timeseries-csv figs/helioswarm_timeseries.csv \
  --timeseries-metadata figs/helioswarm_timeseries_metadata.json \
  --max-order 6 \
  --n-lambda-bins 20 \
  --undersampled-fraction 0.01 \
  --plot \
  --output-dir figs
```

Structure-function analysis directly from a simulation cube:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_structure_functions.py \
  --input-mode simulation_cube \
  --simulation-path /path/to/ot3D_field_75.npy \
  --sim-box-x 314.15926 \
  --sim-box-y 314.15926 \
  --sim-box-z 1570.79632 \
  --max-order 6 \
  --n-lambda-bins 20 \
  --cube-candidate-pairs 20000 \
  --cube-target-pairs-per-bin 256 \
  --cube-random-seed 0 \
  --cube-diagnostics \
  --cube-compare-local-reference \
  --cube-reference-max-offset 1 \
  --plot \
  --output-dir /tmp/sparse_recon_structure_functions_cube
```

Space-time decorrelation analysis from an existing moving-observatory time-series product:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_space_time_correlation.py \
  --timeseries-csv figs/helioswarm_timeseries.csv \
  --timeseries-metadata figs/helioswarm_timeseries_metadata.json \
  --spacecraft-labels H,N1,N2,N3,N4,N5,N6,N7,N8 \
  --n-r-bins 24 \
  --max-tau-seconds 60 \
  --plot \
  --plot-contour \
  --output-dir figs
```

Lag-tetrahedra analysis from saved Elsasser pair products:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_lag_tetrahedra.py \
  --timeseries-metadata figs_velocity/helioswarm_timeseries_metadata.json \
  --elsasser-pairs-npz figs_velocity/helioswarm_timeseries_elsasser_pairs.npz \
  --elsasser-pairs-json figs_velocity/helioswarm_timeseries_elsasser_pairs.json \
  --time-index 0 \
  --dep-max 0.85 \
  --max-arrows 400 \
  --highlight-tetrahedron-index 120 \
  --plot \
  --output-dir figs_lag_tetrahedra
```

## Data Sources

### Synthetic fields

The synthetic field generator supports these benchmark families:

- `smooth`
- `high_frequency`
- `noisy_smooth`
- `smooth_3d`
- `high_frequency_3d`

### Simulation snapshots

Simulation-backed workflows support structured snapshots such as:

- `.npy` vector fields shaped `(nx, ny, nz, 3)`
- `.bp` field snapshots when variable names are supplied
- `.bp` packed ion-moment snapshots for velocity/density workflows

See [data/README.md](/home/kgklein/Codes/sparse-field-reconstruction/data/README.md) for local data expectations.

### HelioSwarm trajectories

HelioSwarm sampling can be driven from either:

- a single `.cdf` file
- a directory containing monthly `.cdf` summary files

Accepted `--hs-time` formats include:

- `YYYY-MM-DD HH`
- `YYYY-MM-DD HH:MM`
- `YYYY-MM-DD HH:MM:SS`
- the same forms with `T` instead of a space

The reader selects the nearest available timestamp in the CDF data.

## Coordinate Systems

Simulation-backed HelioSwarm workflows use two coordinate systems:

- HelioSwarm spacecraft positions start in physical `km`
- simulation boxes are specified in proton gyroradius units `rho_p`

For simulation-backed reconstruction and moving-observatory runs, the current transform pipeline is:

1. Load HelioSwarm positions from CDF files in `km`.
2. Convert them to hub-relative coordinates in `km`.
3. Divide by `--rho-p-km` to convert the formation from `km` to `rho_p`.
4. Use `--sim-box-x`, `--sim-box-y`, and `--sim-box-z` as the full simulation-box lengths in `rho_p`.
5. Translate the HelioSwarm formation so its centroid sits at the center of that simulation box.

This simulation-backed transform is recorded in metadata as `km_to_rho_p_centered_box`.

## Output Products

Common outputs across workflows include:

- baseline benchmark outputs such as `results.jsonl`, per-experiment `metrics.json`, and `overview.png`
- time-series products such as `helioswarm_timeseries.csv` and `helioswarm_timeseries_metadata.json`
- optional Elsasser pair products such as `helioswarm_timeseries_elsasser_pairs.npz` and `helioswarm_timeseries_elsasser_pairs.json`
- structure-function outputs such as `structure_functions.json`, `structure_functions.png`, and optional diagnostics files
- space-time decorrelation outputs such as `space_time_correlation.json` and `space_time_correlation.png`
- lag-tetrahedra outputs such as `lag_tetrahedra.json`, `lag_tetrahedra_ep_scatter.png`, `lag_tetrahedra_yaglom_flux.png`, `lag_tetrahedra_epsilon_diagnostics.png`, `lag_tetrahedra_epsilon_diagnostics_log_km.png`, and `lag_tetrahedra_baseline_projections.png`

The exact inputs and outputs for each run mode are documented in [docs/workflows.md](/home/kgklein/Codes/sparse-field-reconstruction/docs/workflows.md).

## Repository Structure

```text
sparse-field-reconstruction/
|- quick_run*.sh
|- scripts/
|- src/sparse_recon/
|- tests/
|- docs/
`- data/
```

Important directories:

- [`scripts/`](/home/kgklein/Codes/sparse-field-reconstruction/scripts): runnable entrypoints
- [`src/sparse_recon/`](/home/kgklein/Codes/sparse-field-reconstruction/src/sparse_recon): package code for datasets, sampling, methods, analysis, and visualization
- [`tests/`](/home/kgklein/Codes/sparse-field-reconstruction/tests): unit and integration tests
- [`docs/`](/home/kgklein/Codes/sparse-field-reconstruction/docs): workflow documentation

## Current Notes

- Baseline reconstruction methods currently available are `nearest`, `linear`, and `rbf`.
- Moving-observatory time-series runs are simulation-only and require all 9 valid HelioSwarm spacecraft including hub `H`.
- Space-time decorrelation currently requires a spacecraft label list that includes the hub.
- Lag-tetrahedra analysis depends on saved Elsasser pair outputs plus the matching time-series metadata file.
- Large simulation-backed runs can be expensive compared with the quick-run examples.

## Contributing

See [CONTRIBUTING.md](/home/kgklein/Codes/sparse-field-reconstruction/CONTRIBUTING.md) for contribution and testing guidance.

## License

MIT.
