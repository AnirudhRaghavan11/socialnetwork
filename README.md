# Social Network Misinformation Diffusion Model

Network.py has the model and the diffusion, please check that for details. 

The current implementation compares three transparency settings:
- `opaque`
- `partial`
- `full`

and two network types:
- `scale_free`
- `small_world`

The model is designed for the EECE 7065 final project on network transparency and misinformation diffusion.

## Model Summary

Each run generates a synthetic network and initializes agents as a mix of:
- `skeptic`
- `normal`
- believer seeds

Believer-related states are:
- `source_believer`
- `strong_believer`
- `weak_believer`

High-level dynamics:
- A small number of seed nodes start as believers.
- Other agents begin as either `normal` or `skeptic`.
- At each timestep, agents are updated synchronously.
- `skeptic` agents are harder to convince than `normal` agents.
- Newly convinced users become `strong_believer`.
- A capped fraction of strong believers can become `weak_believer` over time.
- `weak_believer` nodes still spread misinformation, but with lower influence.

Transparency conditions affect the effective threshold for adoption:
- `opaque`: no extra structural information
- `partial`: threshold changes based on hop count
- `full`: threshold changes based on hop count and path diversity

## Repository Contents

- [network.py](./network.py): main simulation, metrics, plotting, and experiment pipeline
- [.gitignore](./.gitignore): excludes generated outputs, caches, and virtual environments

Generated outputs are written to:
- `outputs/data/`
- `outputs/plots/`

These are ignored by Git by default.

## Requirements

The script uses:
- Python 3
- `networkx`
- `matplotlib`
- `scipy`
- `pyyaml` (optional, only needed for `config.yaml`)

If you are using a virtual environment, activate it before running the script.

## Running The Model

Run the full pipeline:

```bash
python network.py
```

Run a smaller quick test:

```bash
python network.py --quick
```

Run with manual parameter overrides:

```bash
python network.py --quick --threshold-mean 0.2 --threshold-std 0.08 --seed-count 3
```

Other useful overrides include:

```bash
python network.py \
  --alpha 0.04 \
  --beta 0.08 \
  --skeptic-fraction 0.5 \
  --skeptic-threshold-bias 0.1 \
  --weak-believer-influence 0.4 \
  --weak-believer-fraction-cap 0.3
```

## Key Parameters

The default configuration lives in `DEFAULT_CONFIG` inside [network.py](./network.py).

Important parameters:
- `network_sizes`: network sizes to evaluate
- `runs_per_config`: number of runs per condition
- `seed_count`: number of initial believer seeds
- `alpha`: hop-count transparency effect
- `beta`: path-diversity transparency effect
- `skeptic_fraction`: fraction of agents initialized as skeptics
- `skeptic_threshold_bias`: extra resistance for skeptics
- `weak_believer_influence`: influence weight of weak believers
- `weakening_per_step_min` / `weakening_per_step_max`: range of strong believers weakened per timestep
- `weak_believer_fraction_cap`: maximum fraction of believers allowed to be weak
- `threshold_distribution.mean` / `threshold_distribution.std`: baseline adoption threshold distribution

## Output Files

The pipeline produces:
- per-condition run logs
- per-condition metrics CSVs
- combined metrics/statistics CSVs
- sensitivity-analysis CSVs
- time-series plots
- boxplots of metrics
- cascade-distribution plots

Typical output filenames look like:
- `outputs/data/experiment_opaque_<timestamp>_runs.csv`
- `outputs/data/experiment_full_<timestamp>_metrics.csv`
- `outputs/plots/time_series_scale_free_N500_full_<timestamp>.png`

## Metrics

For each run, the script computes:
- `t25`: timestep when 25% adoption is reached
- `t50`: timestep when 50% adoption is reached
- `peak_infection_ratio`
- `final_adoption_ratio`
- `cascade_size`
- `last_change_timestep`

## Notes For Teammates

- The code currently keeps the model in a single file for simplicity.
- Each run now regenerates a fresh graph, so repeated runs are independent.
- Generated CSVs and plots are not committed by default because they are ignored in `.gitignore`.
- If you want to keep a specific result set, either remove those ignore rules temporarily or copy the files elsewhere.

## Suggested Workflow

1. Pull the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies.
4. Run `python network.py --quick` first.
5. Inspect `outputs/data/` and `outputs/plots/`.
6. Adjust parameters and rerun as needed.

## GitHub

Repository:

`https://github.com/AnirudhRaghavan11/socialnetwork`
