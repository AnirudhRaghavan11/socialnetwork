# Social Network Misinformation Diffusion Model

Network.py contains the simulation model, experiment pipeline, metrics, statistical tests, and plotting utilities.

The current implementation compares three transparency settings:
- `opaque`
- `partial`
- `full`

and two network types:
- `scale_free`
- `small_world`

The model is designed for the EECE 7065 final project on network transparency and misinformation diffusion.

## Model Summary

This implementation uses a heterogeneous-agent diffusion model with one true source of misinformation per run.

Each run generates a synthetic network and initializes agents as a mix of:
- `skeptic`
- `normal`

Believer-related states are:
- `source_believer`
- `strong_believer`
- `weak_believer`

High-level dynamics:
- Exactly one true source node starts as believing.
- All other agents begin as either `normal` or `skeptic`.
- At each timestep, agents are updated synchronously.
- `skeptic` agents are harder to convince than `normal` agents.
- Newly convinced users become `strong_believer`.
- A capped fraction of strong believers can become `weak_believer` over time.
- `weak_believer` nodes still spread misinformation, but with lower influence than strong believers.

## Transparency Signals

The core transparency variables are:

- **Hop count**: distance from the true source along the diffusion path that first led an agent to adopt.
- **Path diversity**: the cumulative number of distinct neighbors through which the content reached an agent before first adoption.

These metadata are assigned at first adoption and then frozen. They are not rewritten later.

Transparency conditions affect the effective threshold for adoption:
- `opaque`: no structural transparency signal
- `partial`: threshold changes based on hop count
- `full`: threshold changes based on hop count and path diversity

## Repository Contents

- [network.py](./network.py): main simulation, metrics, plotting, and experiment pipeline
- [README.md](./README.md): project overview and usage notes
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

## Running the Model

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
python network.py --quick --threshold-mean 0.2 --threshold-std 0.08
```

Other useful overrides include:

```bash
python network.py \
  --alpha 0.04 \
  --beta 0.08 \
  --skeptic-fraction 0.5 \
  --skeptic-threshold-bias 0.1 \
  --weak-believer-influence 0.4 \
  --weak-believer-fraction-cap 0.3 \
  --random-seed 2025
```

## Key Parameters

The default configuration lives in `DEFAULT_CONFIG` inside [network.py](./network.py).

Important parameters:
- `network_sizes`: network sizes to evaluate
- `runs_per_config`: number of runs per condition
- `seed_count`: number of initial believer seeds; the main model uses `1`
- `alpha`: hop-count transparency effect
- `beta`: path-diversity transparency effect
- `random_seed`: master seed for reproducible graph generation and run initialization
- `skeptic_fraction`: fraction of agents initialized as skeptics
- `skeptic_threshold_bias`: extra resistance for skeptic agents
- `weak_believer_influence`: influence weight of weak believers
- `weakening_per_step_min` / `weakening_per_step_max`: range of strong believers weakened per timestep
- `weak_believer_fraction_cap`: maximum fraction of believers allowed to be weak
- `threshold_distribution.mean` / `threshold_distribution.std`: baseline adoption threshold distribution

## Output Files

The pipeline produces:
- per-condition run logs
- per-condition metrics CSVs
- per-condition adoption-level CSVs
- combined metrics/statistics CSVs
- sensitivity-analysis CSVs (run on both scale-free and small-world topologies)
- time-series plots
- boxplots of metrics
- cascade-distribution plots

Typical output filenames look like:
- `outputs/data/experiment_opaque_<timestamp>_runs.csv`
- `outputs/data/experiment_full_<timestamp>_metrics.csv`
- `outputs/data/experiment_partial_<timestamp>_adoptions.csv`
- `outputs/data/adoptions_<label>_<timestamp>.csv`
- `outputs/plots/time_series_scale_free_N500_full_<timestamp>.png`

## Metrics

For each run, the script computes:
- `t25`: timestep when 25% adoption is reached
- `t50`: timestep when 50% adoption is reached
- `peak_infection_ratio`
- `final_adoption_ratio`
- `cascade_size`
- `last_change_timestep`

## Adoption-Level Logging

The pipeline also records first-adoption events in a separate CSV.

For each newly adopting agent, the adoption log records:
- `run_id`
- `graph_seed`
- `init_seed`
- `network_type`
- `network_size`
- `condition`
- `timestep`
- `node_id`
- `old_state`
- `new_state`
- `hop_count_at_adoption`
- `path_diversity_at_adoption`

This file is useful for verifying that hop count and cumulative path diversity are being computed correctly.

## Reproducibility

Each run now uses deterministic seeding for:
- graph generation
- agent initialization
- source selection
- weakening dynamics

If you rerun the pipeline with the same `--random-seed` and the same configuration, the simulation outputs should be reproducible apart from timestamped filenames.

## Notes for Teammates

- The code currently keeps the model in a single file for simplicity.
- Each run regenerates a fresh graph, so repeated runs are independent realizations of the selected network model.
- Generated CSVs and plots are not committed by default because they are ignored in `.gitignore`.
- If you want to keep a specific result set, either remove those ignore rules temporarily or copy the files elsewhere.
- Condition strings are defined as module-level constants: `COND_OPAQUE`, `COND_PARTIAL`, `COND_FULL`, and `ALL_CONDITIONS`. Use these instead of raw strings if you import from or extend `network.py`.
- The hop-count sentinel value is defined as `INF_HOP = 10**9`. Use this constant rather than hardcoding large numbers when checking or initializing hop counts.

## Suggested Workflow

1. Pull the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies.
4. Run `python network.py --quick` first.
5. Inspect `outputs/data/` and `outputs/plots/`.
6. Check the adoption CSVs to verify hop-count and path-diversity behavior.
7. Adjust parameters and rerun as needed.

## GitHub

Repository:

`https://github.com/AnirudhRaghavan11/socialnetwork`
