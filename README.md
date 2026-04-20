# Sparse Field Reconstruction

A benchmarking framework for reconstructing vector fields from sparse spatial measurements.

This project is aimed at comparing reconstruction methods in settings where we know the underlying field, sample it sparsely, reconstruct it over a larger volume, and quantify how well that reconstruction performs. The motivating use case is space plasma physics, where a sparse spacecraft formation samples a much larger 3D region.

## Overview

The current workflow supports both synthetic and simulation-backed fields:

1. load or generate a ground-truth vector field
2. choose sparse sample positions
3. reconstruct the field over the full domain
4. compute common metrics and save diagnostic plots

Supported inputs now include:

- synthetic 2D and 3D benchmark fields
- HelioSwarm-driven 3D sample positions from CDF trajectory files
- dense 3D simulation snapshots stored as `.npy`

The main user entrypoint is [`scripts/run_baseline.py`](/home/kgklein/Codes/sparse-field-reconstruction/scripts/run_baseline.py).

## Setup

```bash
git clone https://github.com/kgklein/sparse-field-reconstruction.git
cd sparse-field-reconstruction

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev,notebooks]"
```

Most example runs in this repo use:

- `PYTHONPATH=src` so the package can be imported without installation issues
- `MPLCONFIGDIR=/tmp/mpl` to give Matplotlib a writable config/cache directory

## Quick Runs

The repo includes small worked examples as shell scripts:

- [`quick_run.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run.sh): 2D synthetic smoke run
- [`quick_run_3d.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_3d.sh): 3D synthetic smoke run
- [`quick_run_3d_hs.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_3d_hs.sh): 3D synthetic field with HelioSwarm sampling
- [`quick_run_3d_sim_hs.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_3d_sim_hs.sh): local example of a 3D simulation snapshot with HelioSwarm sampling; update the data paths for your machine before running it
- [`quick_run_hs_timeseries.sh`](/home/kgklein/Codes/sparse-field-reconstruction/quick_run_hs_timeseries.sh): local example of a moving-observatory HelioSwarm time-series run through a static simulation snapshot

You can run them directly, for example:

```bash
./quick_run.sh
./quick_run_3d.sh
./quick_run_3d_hs.sh
./quick_run_3d_sim_hs.sh
./quick_run_hs_timeseries.sh
```

You can also run the main script directly.

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
  --rho-p-km 50 \
  --sim-box-x 20 \
  --sim-box-y 20 \
  --sim-box-z 20 \
  --include-hub \
  --output-dir /tmp/sparse_recon_sim_hs
```

Moving-observatory HelioSwarm time-series run:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl python3 scripts/run_hs_timeseries.py \
  --simulation-path /path/to/ot3D_field_75.npy \
  --hs-path /path/to/HelioSwarm/DRM \
  --hs-time "2029-09-26 00:00:00" \
  --rho-p-km 50 \
  --sim-box-x 20 \
  --sim-box-y 20 \
  --sim-box-z 20 \
  --vx-kms 15 \
  --vy-kms 0 \
  --vz-kms -5 \
  --dt-seconds 1.0 \
  --n-steps 120 \
  --plot-timeseries \
  --output-dir /tmp/sparse_recon_hs_timeseries
```

## Data Sources

### Synthetic fields

The synthetic field generator currently supports a small set of benchmark field families, including:

- `smooth`
- `high_frequency`
- `noisy_smooth`
- `smooth_3d`
- `high_frequency_3d`

These are the easiest way to validate the pipeline and compare methods quickly.

### Simulation snapshots

Simulation-backed runs currently support dense `.npy` snapshots with shape:

```text
(nx, ny, nz, 3)
```

The loader interprets these as a structured 3D vector field with:

- 3 vector components in the last dimension
- a uniform Cartesian grid
- axes currently normalized to a unit box `[0, 1]^3`

The current real example used in development is:

```text
/home/kgklein/Codes/gkeyll/data/ot3D_field_75.npy
```

### HelioSwarm trajectories

HelioSwarm sampling can be driven from either:

- a single `.cdf` file
- a directory containing monthly `.cdf` summary files

The current reader uses:

- `Epoch`
- `Position`
- `Spacecraft_Label`

Accepted `--hs-time` input formats include:

- `YYYY-MM-DD HH`
- `YYYY-MM-DD HH:MM`
- `YYYY-MM-DD HH:MM:SS`
- the same forms with `T` instead of a space

The reader selects the nearest available timestamp in the CDF data.

Notes for current HelioSwarm data:

- some DRM directories contain an empty placeholder file such as `hsconcept_l2-summary_00000000_v0.1.0.cdf`
- DRM `v0.1.0` files can include an `N/A` placeholder slot in `Spacecraft_Label`
- the loader skips empty CDFs and ignores that placeholder slot automatically

## Main Script

The main driver script is:

```bash
python3 scripts/run_baseline.py ...
```

The moving-observatory time-series driver is:

```bash
python3 scripts/run_hs_timeseries.py ...
```

Important arguments:

- `--data-source`: choose `synthetic` or `simulation`
- `--field-kind`: synthetic field family such as `smooth` or `smooth_3d`
- `--simulation-path`: path to a `.npy` simulation snapshot when using `--data-source simulation`
- `--hs-path`: HelioSwarm `.cdf` file or directory of `.cdf` files
- `--hs-time`: requested HelioSwarm timestamp; nearest available sample is used
- `--rho-p-km`: proton gyroradius in km for simulation-backed HelioSwarm runs
- `--sim-box-x`, `--sim-box-y`, `--sim-box-z`: full simulation-box lengths in units of `rho_p` for simulation-backed HelioSwarm runs
- `--include-hub`: include the hub spacecraft in HelioSwarm-driven sampling
- `--output-dir`: directory for metrics and plots

Useful defaults:

- if `--data-source` is omitted, the script uses synthetic data
- HelioSwarm sampling is only activated when both `--hs-path` and `--hs-time` are provided
- simulation-backed HelioSwarm runs also require `--rho-p-km` and all three `--sim-box-*` flags
- `--include-hub` only matters for HelioSwarm-backed runs
- baseline methods currently available are `nearest`, `linear`, and `rbf`
- moving-observatory time-series runs are simulation-only and always require all 9 valid HelioSwarm spacecraft

## Outputs

Each experiment writes a subdirectory under the chosen output directory.

Common outputs:

- `metrics.json`: metrics and run metadata for one experiment
- `results.jsonl`: line-delimited summary records across all experiments in the run
- `overview.png`: main reconstruction diagnostic plot

Additional HelioSwarm outputs:

- `helioswarm_physical.png`: spacecraft formation in physical hub-relative coordinates
- `helioswarm_scaled.png`: spacecraft formation after scaling into the reconstruction box, or in simulation coordinates (`rho_p`) for simulation-backed HelioSwarm runs
- `helioswarm_timeseries.csv`: moving-observatory time-series table with positions and sampled `Bx`, `By`, `Bz`
- `helioswarm_timeseries_metadata.json`: metadata for the moving-observatory run, including transform and velocity details
- `helioswarm_timeseries.png`: optional three-panel `Bx`/`By`/`Bz` line plot for all 9 spacecraft

## Repository Structure

```text
sparse-field-reconstruction/
├── quick_run*.sh
├── scripts/
├── src/sparse_recon/
├── tests/
└── data/
```

Important directories:

- [`scripts/`](/home/kgklein/Codes/sparse-field-reconstruction/scripts): runnable experiment entrypoints
- [`src/sparse_recon/`](/home/kgklein/Codes/sparse-field-reconstruction/src/sparse_recon): datasets, sampling, methods, metrics, pipeline, and visualization
- [`tests/`](/home/kgklein/Codes/sparse-field-reconstruction/tests): unit and integration tests

## Current Limitations / Notes

- simulation `.npy` snapshots are currently assumed to live on a uniform unit box `[0, 1]^3`
- only single-snapshot simulation files are supported right now
- HelioSwarm support currently uses static snapshot sampling, not full time-evolving trajectories
- moving-observatory HelioSwarm time-series runs currently sample a static simulation snapshot while translating the observatory rigidly with periodic wrapping
- large 3D simulation reconstructions can be expensive, especially for full `448^3` query volumes
- for large real-data runs, a full reconstruction may take significantly longer than the small quick-run examples
- the baseline methods are `nearest`, `linear`, and `rbf`; `rbf` is the most practical default for current demos
- the 3D overview plot now renders sample positions across all panels and distinguishes the hub when HelioSwarm metadata is present

## Contributing

For any new method or dataset path:

- test on a shared dataset
- compare against a baseline
- report common metrics
- save outputs reproducibly

## License

MIT.
