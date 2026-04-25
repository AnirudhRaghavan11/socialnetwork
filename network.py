import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".mplconfig")

import matplotlib.pyplot as plt
import networkx as nx

try:
    import yaml
except ImportError:
    yaml = None

try:
    from scipy.stats import mannwhitneyu
except ImportError:
    mannwhitneyu = None


os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/data", exist_ok=True)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)


DEFAULT_CONFIG = {
    "network_sizes": [500, 1000],
    "runs_per_config": 100,
    "seed_count": 1,
    "max_timesteps": 200,
    "conditions": ["opaque", "partial", "full"],
    "alpha": 0.1,
    "beta": 0.05,
    "random_seed": 12345,
    "skeptic_fraction": 0.5,
    "skeptic_threshold_bias": 0.05,
    "weak_believer_influence": 0.4,
    "weakening_per_step_min": 1,
    "weakening_per_step_max": 3,
    "weak_believer_fraction_cap": 0.3,
    "threshold_distribution": {"mean": 0.15, "std": 0.05},
    "network_params": {
        "scale_free": {"m": 3},
        "small_world": {"k": 6, "p": 0.3},
    },
    "size_overrides": {},
}


@dataclass
class Agent:
    id: int
    threshold: float
    state: str
    hop_count: int
    path_diversity: int
    neighbors: list
    exposure_sources: set

    def __post_init__(self):
        self._candidate_hop = self.hop_count
        self._candidate_path_diversity = self.path_diversity
        self._candidate_exposure_sources = set(self.exposure_sources)

    def compute_next_state(
        self,
        condition,
        neighbor_states,
        alpha=0.1,
        beta=0.05,
        skeptic_threshold_bias=0.05,
        weak_believer_influence=0.4,
    ):
        believer_states = {"source_believer", "strong_believer", "weak_believer"}

        # Once an agent is believing, its adoption metadata are frozen.
        if self.state in believer_states:
            self._candidate_hop = self.hop_count
            self._candidate_path_diversity = self.path_diversity
            self._candidate_exposure_sources = set(self.exposure_sources)
            return self.state

        if not self.neighbors:
            self._candidate_hop = self.hop_count
            self._candidate_path_diversity = self.path_diversity
            self._candidate_exposure_sources = set(self.exposure_sources)
            return self.state

        sharing = [n for n in neighbor_states if n["state"] in believer_states]

        # No current exposure this step: keep existing exposure history.
        if not sharing:
            self._candidate_hop = self.hop_count
            self._candidate_path_diversity = len(self.exposure_sources)
            self._candidate_exposure_sources = set(self.exposure_sources)
            return self.state

        influence = 0.0
        for neighbor in sharing:
            if neighbor["state"] == "weak_believer":
                influence += weak_believer_influence
            else:
                influence += 1.0
        fraction = influence / len(self.neighbors)

        sharing_ids = {n["id"] for n in sharing}
        candidate_exposure_sources = self.exposure_sources.union(sharing_ids)
        self._candidate_exposure_sources = candidate_exposure_sources
        self._candidate_path_diversity = len(candidate_exposure_sources)

        finite_hops = [n["hop_count"] for n in sharing if n["hop_count"] < 10**8]
        if finite_hops:
            self._candidate_hop = min(finite_hops) + 1
        else:
            self._candidate_hop = self.hop_count

        state_bias = skeptic_threshold_bias if self.state == "skeptic" else 0.0

        if condition == "opaque":
            effective_threshold = self.threshold + state_bias
        elif condition == "partial":
            effective_threshold = self.threshold + state_bias + alpha * self._candidate_hop
        elif condition == "full":
            effective_threshold = (
                self.threshold
                + state_bias
                + alpha * self._candidate_hop
                - beta * self._candidate_path_diversity
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

        effective_threshold = max(0.0, min(1.0, effective_threshold))

        if fraction > effective_threshold:
            return "strong_believer"

        return self.state


def load_config(config_path="config.yaml"):
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(config_path) and yaml is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        config = deep_merge(config, user_cfg)
    elif os.path.exists(config_path) and yaml is None:
        print(f"Warning: {config_path} exists but PyYAML is not installed; using built-in defaults.")
    return config


def deep_merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def get_config_for_size(config, network_size):
    size_config = deep_merge(config, {})
    override = config.get("size_overrides", {}).get(str(network_size))
    if override is None:
        override = config.get("size_overrides", {}).get(network_size)
    if override:
        size_config = deep_merge(size_config, override)
    return size_config


def generate_network(network_type, n, params, seed=None):
    if network_type == "scale_free":
        m = params.get("m", 3)
        return nx.barabasi_albert_graph(n, m, seed=seed)
    if network_type == "small_world":
        k = params.get("k", 6)
        p = params.get("p", 0.3)
        return nx.watts_strogatz_graph(n, k, p, seed=seed)
    raise ValueError(f"Invalid network type: {network_type}")


def initialize_agents(graph, threshold_mean, threshold_std, skeptic_fraction=0.5, rng=None):
    if rng is None:
        rng = random.Random()

    agents = {}
    for node in graph.nodes:
        threshold = max(0.0, min(1.0, rng.gauss(threshold_mean, threshold_std)))
        initial_state = "skeptic" if rng.random() < skeptic_fraction else "normal"
        agents[node] = Agent(
            id=node,
            threshold=threshold,
            state=initial_state,
            hop_count=10**9,
            path_diversity=0,
            neighbors=list(graph.neighbors(node)),
            exposure_sources=set(),
        )
    return agents


def count_states(agents):
    num_skeptic = sum(1 for a in agents.values() if a.state == "skeptic")
    num_normal = sum(1 for a in agents.values() if a.state == "normal")
    num_source = sum(1 for a in agents.values() if a.state == "source_believer")
    num_strong = sum(1 for a in agents.values() if a.state == "strong_believer")
    num_weak = sum(1 for a in agents.values() if a.state == "weak_believer")
    num_believing = num_source + num_strong + num_weak
    return num_skeptic, num_normal, num_source, num_strong, num_weak, num_believing


def weaken_believers(
    agents,
    weakening_per_step_min=1,
    weakening_per_step_max=3,
    weak_believer_fraction_cap=0.3,
    rng=None,
):
    if rng is None:
        rng = random.Random()

    strong_believers = [agent for agent in agents.values() if agent.state == "strong_believer"]
    weak_believers = [agent for agent in agents.values() if agent.state == "weak_believer"]
    source_believers = [agent for agent in agents.values() if agent.state == "source_believer"]
    total_believers = len(source_believers) + len(strong_believers) + len(weak_believers)

    if not strong_believers or total_believers == 0:
        return 0

    max_weak_allowed = math.floor(weak_believer_fraction_cap * total_believers)
    remaining_slots = max_weak_allowed - len(weak_believers)
    if remaining_slots <= 0:
        return 0

    num_to_weaken = rng.randint(weakening_per_step_min, weakening_per_step_max)
    num_to_weaken = min(num_to_weaken, remaining_slots, len(strong_believers))
    if num_to_weaken <= 0:
        return 0

    for agent in rng.sample(strong_believers, num_to_weaken):
        agent.state = "weak_believer"

    return num_to_weaken


def run_single_simulation(
    graph,
    network_type,
    network_size,
    condition,
    run_id,
    max_timesteps,
    alpha,
    beta,
    skeptic_fraction,
    skeptic_threshold_bias,
    weak_believer_influence,
    weakening_per_step_min,
    weakening_per_step_max,
    weak_believer_fraction_cap,
    threshold_mean,
    threshold_std,
    seed_count=1,
    graph_seed=None,
    init_seed=None,
):
    believer_states = {"source_believer", "strong_believer", "weak_believer"}
    debug_run = (run_id == 0)

    rng = random.Random(init_seed)

    agents = initialize_agents(
        graph,
        threshold_mean,
        threshold_std,
        skeptic_fraction,
        rng=rng,
    )

    if seed_count != 1:
        raise ValueError("This revised model requires exactly one true source (seed_count=1).")

    seed = rng.choice(list(graph.nodes))
    agents[seed].state = "source_believer"
    agents[seed].hop_count = 0
    agents[seed].path_diversity = 0
    agents[seed].exposure_sources = set()

    if debug_run:
        print("DEBUG timestep 0")
        print("source node:", seed)
        print("source state:", agents[seed].state)
        print("source hop_count:", agents[seed].hop_count)
        print("source path_diversity:", agents[seed].path_diversity)

        skeptic, normal, source, strong, weak, believing = count_states(agents)
        print({
            "skeptic": skeptic,
            "normal": normal,
            "source_believer": source,
            "strong_believer": strong,
            "weak_believer": weak,
            "believing_total": believing,
        })

    logs = []
    history = []
    last_change_timestep = 0
    first_adoption_record = {}

    skeptic, normal, source, strong, weak, believing = count_states(agents)
    logs.append(
        {
            "run_id": run_id,
            "graph_seed": graph_seed,
            "init_seed": init_seed,
            "network_size": network_size,
            "network_type": network_type,
            "condition": condition,
            "timestep": 0,
            "num_new_believing": believing,
            "num_skeptic": skeptic,
            "num_normal": normal,
            "num_source_believer": source,
            "num_strong_believer": strong,
            "num_weak_believer": weak,
            "num_weakened_this_step": 0,
            "num_believing": believing,
        }
    )
    history.append(believing)

    for t in range(1, max_timesteps + 1):
        next_states = {}
        next_hops = {}
        next_diversity = {}
        next_exposure_sources = {}

        for agent in agents.values():
            neighbor_states = [
                {
                    "id": nid,
                    "state": agents[nid].state,
                    "hop_count": agents[nid].hop_count,
                }
                for nid in agent.neighbors
            ]

            next_state = agent.compute_next_state(
                condition=condition,
                neighbor_states=neighbor_states,
                alpha=alpha,
                beta=beta,
                skeptic_threshold_bias=skeptic_threshold_bias,
                weak_believer_influence=weak_believer_influence,
            )

            next_states[agent.id] = next_state
            next_hops[agent.id] = agent._candidate_hop
            next_diversity[agent.id] = agent._candidate_path_diversity
            next_exposure_sources[agent.id] = set(agent._candidate_exposure_sources)

        for agent in agents.values():
            old_state = agent.state
            new_state = next_states[agent.id]

            was_believer = old_state in believer_states
            will_believe = new_state in believer_states

            agent.state = new_state

            if (not was_believer) and will_believe:
                agent.hop_count = next_hops[agent.id]
                agent.path_diversity = next_diversity[agent.id]
                agent.exposure_sources = set(next_exposure_sources[agent.id])

                if agent.id not in first_adoption_record:
                    first_adoption_record[agent.id] = (
                        next_hops[agent.id],
                        next_diversity[agent.id],
                    )

                if debug_run:
                    print(
                        "ADOPT",
                        {
                            "timestep": t,
                            "node_id": agent.id,
                            "old_state": old_state,
                            "new_state": new_state,
                            "hop_count_at_adoption": next_hops[agent.id],
                            "path_diversity_at_adoption": next_diversity[agent.id],
                            "exposure_sources_size": len(next_exposure_sources[agent.id]),
                        }
                    )

            elif (not was_believer) and (not will_believe):
                old_size = len(agent.exposure_sources)
                agent.exposure_sources = set(next_exposure_sources[agent.id])
                new_size = len(agent.exposure_sources)

                if debug_run and new_size > old_size:
                    print(
                        "EXPOSURE_ACCUMULATED",
                        {
                            "timestep": t,
                            "node_id": agent.id,
                            "old_size": old_size,
                            "new_size": new_size,
                            "current_exposure_sources": sorted(list(agent.exposure_sources)),
                        }
                    )

        if debug_run:
            for node_id, (first_hop, first_div) in first_adoption_record.items():
                a = agents[node_id]
                if a.hop_count != first_hop or a.path_diversity != first_div:
                    print(
                        "ERROR: frozen metadata changed",
                        {
                            "node_id": node_id,
                            "expected": (first_hop, first_div),
                            "actual": (a.hop_count, a.path_diversity),
                        }
                    )

        num_weakened = weaken_believers(
            agents,
            weakening_per_step_min=weakening_per_step_min,
            weakening_per_step_max=weakening_per_step_max,
            weak_believer_fraction_cap=weak_believer_fraction_cap,
            rng=rng,
        )

        previous_believing = history[-1]
        skeptic, normal, source, strong, weak, believing = count_states(agents)
        new_believing = believing - previous_believing
        if new_believing > 0:
            last_change_timestep = t

        logs.append(
            {
                "run_id": run_id,
                "graph_seed": graph_seed,
                "init_seed": init_seed,
                "network_size": network_size,
                "network_type": network_type,
                "condition": condition,
                "timestep": t,
                "num_new_believing": new_believing,
                "num_skeptic": skeptic,
                "num_normal": normal,
                "num_source_believer": source,
                "num_strong_believer": strong,
                "num_weak_believer": weak,
                "num_weakened_this_step": num_weakened,
                "num_believing": believing,
            }
        )
        history.append(believing)

        if new_believing == 0 and t > 5:
            for rest_t in range(t + 1, max_timesteps + 1):
                logs.append(
                    {
                        "run_id": run_id,
                        "graph_seed": graph_seed,
                        "init_seed": init_seed,
                        "network_size": network_size,
                        "network_type": network_type,
                        "condition": condition,
                        "timestep": rest_t,
                        "num_new_believing": 0,
                        "num_skeptic": skeptic,
                        "num_normal": normal,
                        "num_source_believer": source,
                        "num_strong_believer": strong,
                        "num_weak_believer": weak,
                        "num_weakened_this_step": 0,
                        "num_believing": believing,
                    }
                )
                history.append(believing)
            break

        if believing == len(agents):
            for rest_t in range(t + 1, max_timesteps + 1):
                logs.append(
                    {
                        "run_id": run_id,
                        "graph_seed": graph_seed,
                        "init_seed": init_seed,
                        "network_size": network_size,
                        "network_type": network_type,
                        "condition": condition,
                        "timestep": rest_t,
                        "num_new_believing": 0,
                        "num_skeptic": 0,
                        "num_normal": 0,
                        "num_source_believer": 1,
                        "num_strong_believer": 0,
                        "num_weak_believer": len(agents) - 1,
                        "num_weakened_this_step": 0,
                        "num_believing": len(agents),
                    }
                )
                history.append(len(agents))
            break

    return logs, history, last_change_timestep


def run_experiment(
    network_type,
    network_size,
    condition,
    network_params,
    num_runs=100,
    max_timesteps=200,
    alpha=0.1,
    beta=0.05,
    skeptic_fraction=0.5,
    skeptic_threshold_bias=0.15,
    weak_believer_influence=0.4,
    weakening_per_step_min=1,
    weakening_per_step_max=3,
    weak_believer_fraction_cap=0.3,
    threshold_mean=0.25,
    threshold_std=0.1,
    seed_count=1,
    base_seed=0,
):
    raw_logs = []
    histories = []
    last_change_timesteps = {}

    for run_id in range(num_runs):
        graph_seed = base_seed + 100_000 + run_id
        init_seed = base_seed + 200_000 + run_id

        graph = generate_network(
            network_type,
            network_size,
            network_params,
            seed=graph_seed,
        )

        run_log, history, last_change_timestep = run_single_simulation(
            graph=graph,
            network_type=network_type,
            network_size=network_size,
            condition=condition,
            run_id=run_id,
            max_timesteps=max_timesteps,
            alpha=alpha,
            beta=beta,
            skeptic_fraction=skeptic_fraction,
            skeptic_threshold_bias=skeptic_threshold_bias,
            weak_believer_influence=weak_believer_influence,
            weakening_per_step_min=weakening_per_step_min,
            weakening_per_step_max=weakening_per_step_max,
            weak_believer_fraction_cap=weak_believer_fraction_cap,
            threshold_mean=threshold_mean,
            threshold_std=threshold_std,
            seed_count=seed_count,
            graph_seed=graph_seed,
            init_seed=init_seed,
        )

        raw_logs.extend(run_log)
        histories.append(history)
        last_change_timesteps[run_id] = last_change_timestep

    return raw_logs, histories, last_change_timesteps


def compute_metrics(run_log, last_change_timestep=None):
    if not run_log:
        return {}

    network_size = run_log[0]["network_size"]
    believers = [row["num_believing"] for row in run_log]

    t25 = None
    t50 = None
    for row in run_log:
        ratio = row["num_believing"] / network_size
        if t25 is None and ratio >= 0.25:
            t25 = row["timestep"]
        if t50 is None and ratio >= 0.50:
            t50 = row["timestep"]

    peak_ratio = max(believers) / network_size
    final_ratio = believers[-1] / network_size
    cascade_size = believers[-1]

    return {
        "run_id": run_log[0]["run_id"],
        "graph_seed": run_log[0].get("graph_seed"),
        "init_seed": run_log[0].get("init_seed"),
        "network_size": run_log[0]["network_size"],
        "network_type": run_log[0]["network_type"],
        "condition": run_log[0]["condition"],
        "t25": t25,
        "t50": t50,
        "peak_infection_ratio": peak_ratio,
        "final_adoption_ratio": final_ratio,
        "cascade_size": cascade_size,
        "last_change_timestep": last_change_timestep,
    }


def group_logs_by_run(raw_logs):
    runs = {}
    for row in raw_logs:
        key = (
            row["network_size"],
            row["network_type"],
            row["condition"],
            row["run_id"],
        )
        runs.setdefault(key, []).append(row)
    for key in runs:
        runs[key] = sorted(runs[key], key=lambda r: r["timestep"])
    return runs


def compute_metrics_for_experiment(raw_logs, last_change_timesteps=None):
    run_groups = group_logs_by_run(raw_logs)
    metrics = []
    for rows in run_groups.values():
        run_id = rows[0]["run_id"]
        last_change_timestep = None
        if last_change_timesteps is not None:
            last_change_timestep = last_change_timesteps.get(run_id)
        metrics.append(compute_metrics(rows, last_change_timestep=last_change_timestep))
    return metrics


def safe_mean(values):
    if not values:
        return math.nan
    return sum(values) / len(values)


def safe_std(values):
    if len(values) < 2:
        return 0.0
    mean = safe_mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def mann_whitney_comparisons(metrics_rows, network_type, network_size):
    if mannwhitneyu is None:
        return []

    pairs = [("opaque", "partial"), ("opaque", "full"), ("partial", "full")]
    metric_names = [
        "t25",
        "t50",
        "peak_infection_ratio",
        "final_adoption_ratio",
        "cascade_size",
    ]

    subset = [
        m
        for m in metrics_rows
        if m["network_type"] == network_type and m["network_size"] == network_size
    ]
    comparisons = []

    for metric in metric_names:
        for c1, c2 in pairs:
            x = [m[metric] for m in subset if m["condition"] == c1 and m[metric] is not None]
            y = [m[metric] for m in subset if m["condition"] == c2 and m[metric] is not None]

            if len(x) == 0 or len(y) == 0:
                comparisons.append(
                    {
                        "network_type": network_type,
                        "network_size": network_size,
                        "metric": metric,
                        "condition_a": c1,
                        "condition_b": c2,
                        "n_a": len(x),
                        "n_b": len(y),
                        "u_stat": math.nan,
                        "p_value": math.nan,
                        "effect_size_rbc": math.nan,
                        "significant_0_05": False,
                    }
                )
                continue

            u_stat, p_value = mannwhitneyu(x, y, alternative="two-sided")
            effect_size_rbc = 1 - (2 * u_stat) / (len(x) * len(y))

            comparisons.append(
                {
                    "network_type": network_type,
                    "network_size": network_size,
                    "metric": metric,
                    "condition_a": c1,
                    "condition_b": c2,
                    "n_a": len(x),
                    "n_b": len(y),
                    "u_stat": u_stat,
                    "p_value": p_value,
                    "effect_size_rbc": effect_size_rbc,
                    "significant_0_05": p_value < 0.05,
                }
            )

    return comparisons


def save_dict_rows(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_dict_rows(rows, path):
    if not rows:
        return

    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def plot_time_series(histories_by_condition, output_path, title):
    plt.figure(figsize=(9, 5))
    for condition, histories in histories_by_condition.items():
        if not histories:
            continue
        num_steps = len(histories[0])
        avg = []
        lower_ci = []
        upper_ci = []
        for t in range(num_steps):
            vals = [h[t] for h in histories]
            mean = safe_mean(vals)
            # Approximate 95% confidence interval around the mean believing curve.
            margin = 1.96 * safe_std(vals) / math.sqrt(len(vals))
            avg.append(mean)
            lower_ci.append(max(0.0, mean - margin))
            upper_ci.append(mean + margin)

        x = list(range(num_steps))
        plt.plot(x, avg, label=f"{condition} avg", linewidth=2)
        plt.fill_between(x, lower_ci, upper_ci, alpha=0.15)

    plt.title(title)
    plt.xlabel("Timestep")
    plt.ylabel("Believing agents")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric_boxplots(metrics_rows, network_type, network_size, output_path):
    metric_names = ["t25", "t50", "peak_infection_ratio", "final_adoption_ratio", "cascade_size"]
    conditions = ["opaque", "partial", "full"]

    subset = [
        m
        for m in metrics_rows
        if m["network_type"] == network_type and m["network_size"] == network_size
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for idx, metric in enumerate(metric_names):
        data = []
        labels = []
        for c in conditions:
            vals = [m[metric] for m in subset if m["condition"] == c and m[metric] is not None]
            if vals:
                data.append(vals)
                labels.append(c)
        if data:
            axes[idx].boxplot(data, labels=labels)
        axes[idx].set_title(metric)
        axes[idx].grid(alpha=0.2)

    axes[-1].axis("off")
    fig.suptitle(f"Metrics by condition ({network_type}, N={network_size})", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cascade_distribution(metrics_rows, network_type, network_size, output_path):
    conditions = ["opaque", "partial", "full"]
    plt.figure(figsize=(9, 5))
    subset = [
        m
        for m in metrics_rows
        if m["network_type"] == network_type and m["network_size"] == network_size
    ]

    for c in conditions:
        vals = [int(m["cascade_size"]) for m in subset if m["condition"] == c]
        if not vals:
            continue
        counts = {}
        for val in vals:
            counts[val] = counts.get(val, 0) + 1
        x = sorted(counts.keys())
        y = [counts[val] / len(vals) for val in x]
        plt.plot(x, y, marker="o", linewidth=1.8, label=c)

    plt.title(f"Cascade size distribution ({network_type}, N={network_size})")
    plt.xlabel("Cascade size")
    plt.ylabel("Share of runs")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_sensitivity_analysis(base_config, network_type="scale_free"):
    alpha_values = [0.05, 0.1, 0.15]
    beta_values = [0.02, 0.05, 0.1]
    threshold_means = [0.05, 0.10, 0.15, 0.20, 0.25]
    output = []

    network_size = base_config["network_sizes"][0]
    base_seed = base_config.get("random_seed", 12345)
    combo_idx = 0

    for alpha in alpha_values:
        for beta in beta_values:
            for t_mean in threshold_means:
                combo_seed = base_seed + combo_idx * 1_000_000

                raw_logs, _, last_change_timesteps = run_experiment(
                    network_type=network_type,
                    network_size=network_size,
                    condition="full",
                    network_params=base_config["network_params"][network_type],
                    num_runs=10,
                    max_timesteps=base_config["max_timesteps"],
                    alpha=alpha,
                    beta=beta,
                    skeptic_fraction=base_config["skeptic_fraction"],
                    skeptic_threshold_bias=base_config["skeptic_threshold_bias"],
                    weak_believer_influence=base_config["weak_believer_influence"],
                    weakening_per_step_min=base_config["weakening_per_step_min"],
                    weakening_per_step_max=base_config["weakening_per_step_max"],
                    weak_believer_fraction_cap=base_config["weak_believer_fraction_cap"],
                    threshold_mean=t_mean,
                    threshold_std=base_config["threshold_distribution"]["std"],
                    seed_count=base_config["seed_count"],
                    base_seed=combo_seed,
                )
                metrics_rows = compute_metrics_for_experiment(raw_logs, last_change_timesteps)
                final_ratios = [m["final_adoption_ratio"] for m in metrics_rows]
                output.append(
                    {
                        "network_size": network_size,
                        "network_type": network_type,
                        "alpha": alpha,
                        "beta": beta,
                        "threshold_mean": t_mean,
                        "avg_final_adoption_ratio": safe_mean(final_ratios),
                    }
                )
                combo_idx += 1

    return output


def run_pipeline(config, label):
    timestamp = datetime.now().strftime("%H%M%S")
    all_raw_logs = []
    all_metrics = []
    all_stats = []

    master_seed = config.get("random_seed", 12345)
    config_counter = 0

    for n in config["network_sizes"]:
        size_config = get_config_for_size(config, n)
        for network_type in ["scale_free", "small_world"]:
            histories_by_condition = {}

            for condition in config["conditions"]:
                config_seed = master_seed + config_counter * 1_000_000

                raw_logs, histories, last_change_timesteps = run_experiment(
                    network_type=network_type,
                    network_size=n,
                    condition=condition,
                    network_params=size_config["network_params"][network_type],
                    num_runs=size_config["runs_per_config"],
                    max_timesteps=size_config["max_timesteps"],
                    alpha=size_config["alpha"],
                    beta=size_config["beta"],
                    skeptic_fraction=size_config["skeptic_fraction"],
                    skeptic_threshold_bias=size_config["skeptic_threshold_bias"],
                    weak_believer_influence=size_config["weak_believer_influence"],
                    weakening_per_step_min=size_config["weakening_per_step_min"],
                    weakening_per_step_max=size_config["weakening_per_step_max"],
                    weak_believer_fraction_cap=size_config["weak_believer_fraction_cap"],
                    threshold_mean=size_config["threshold_distribution"]["mean"],
                    threshold_std=size_config["threshold_distribution"]["std"],
                    seed_count=size_config["seed_count"],
                    base_seed=config_seed,
                )
                all_raw_logs.extend(raw_logs)
                condition_metrics = compute_metrics_for_experiment(raw_logs, last_change_timesteps)
                all_metrics.extend(condition_metrics)
                histories_by_condition[condition] = histories

                runs_path = f"outputs/data/experiment_{condition}_{timestamp}_runs.csv"
                metrics_path = f"outputs/data/experiment_{condition}_{timestamp}_metrics.csv"
                append_dict_rows(raw_logs, runs_path)
                append_dict_rows(condition_metrics, metrics_path)

                config_counter += 1

            network_title = f"N={n}, network={network_type}"
            ts_plot = f"outputs/plots/time_series_{network_type}_N{n}_{label}_{timestamp}.png"
            plot_time_series(histories_by_condition, ts_plot, f"Average believing curve ({network_title})")

            box_plot = f"outputs/plots/boxplot_{network_type}_N{n}_{label}_{timestamp}.png"
            subset_metrics = [
                m
                for m in all_metrics
                if m["network_type"] == network_type and m["network_size"] == n
            ]
            plot_metric_boxplots(subset_metrics, network_type, n, box_plot)

            casc_plot = f"outputs/plots/cascade_{network_type}_N{n}_{label}_{timestamp}.png"
            plot_cascade_distribution(subset_metrics, network_type, n, casc_plot)

            all_stats.extend(mann_whitney_comparisons(all_metrics, network_type, n))

    raw_path = f"outputs/data/simulation_logs_{label}_{timestamp}.csv"
    metrics_path = f"outputs/data/metrics_results_{label}_{timestamp}.csv"
    stats_path = f"outputs/data/statistics_{label}_{timestamp}.csv"
    sens_path = f"outputs/data/sensitivity_{label}_{timestamp}.csv"

    save_dict_rows(all_raw_logs, raw_path)
    save_dict_rows(all_metrics, metrics_path)
    save_dict_rows(all_stats, stats_path)

    sensitivity_rows = run_sensitivity_analysis(config, network_type="scale_free")
    save_dict_rows(sensitivity_rows, sens_path)

    print("Saved raw logs:", raw_path)
    print("Saved metrics:", metrics_path)
    if mannwhitneyu is None:
        print("SciPy not installed; skipped Mann-Whitney tests.")
    else:
        print("Saved statistical tests:", stats_path)
    print("Saved sensitivity summary:", sens_path)


def apply_quick_mode(config):
    quick = deep_merge(config, {})
    quick["network_sizes"] = [200]
    quick["runs_per_config"] = 12
    quick["max_timesteps"] = 60
    return quick


def apply_cli_overrides(config, args):
    updated = deep_merge(config, {})
    if args.random_seed is not None:
        updated["random_seed"] = args.random_seed
    if args.network_sizes:
        updated["network_sizes"] = args.network_sizes
    if args.runs_per_config is not None:
        updated["runs_per_config"] = args.runs_per_config
    if args.seed_count is not None:
        updated["seed_count"] = args.seed_count
    if args.max_timesteps is not None:
        updated["max_timesteps"] = args.max_timesteps
    if args.alpha is not None:
        updated["alpha"] = args.alpha
    if args.beta is not None:
        updated["beta"] = args.beta
    if args.skeptic_fraction is not None:
        updated["skeptic_fraction"] = args.skeptic_fraction
    if args.skeptic_threshold_bias is not None:
        updated["skeptic_threshold_bias"] = args.skeptic_threshold_bias
    if args.weak_believer_influence is not None:
        updated["weak_believer_influence"] = args.weak_believer_influence
    if args.weakening_per_step_min is not None:
        updated["weakening_per_step_min"] = args.weakening_per_step_min
    if args.weakening_per_step_max is not None:
        updated["weakening_per_step_max"] = args.weakening_per_step_max
    if args.weak_believer_fraction_cap is not None:
        updated["weak_believer_fraction_cap"] = args.weak_believer_fraction_cap
    if args.threshold_mean is not None:
        updated["threshold_distribution"]["mean"] = args.threshold_mean
    if args.threshold_std is not None:
        updated["threshold_distribution"]["std"] = args.threshold_std
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Misinformation transparency simulation pipeline")
    parser.add_argument("--random-seed", type=int, help="Override master random seed")
    parser.add_argument("--config", default="config.yaml", help="Path to config yaml file")
    parser.add_argument("--quick", action="store_true", help="Run a faster draft experiment")
    parser.add_argument("--network-sizes", nargs="+", type=int, help="Override network sizes")
    parser.add_argument("--runs-per-config", type=int, help="Override runs per configuration")
    parser.add_argument("--seed-count", type=int, help="Override number of initial seed nodes")
    parser.add_argument("--max-timesteps", type=int, help="Override max timesteps")
    parser.add_argument("--alpha", type=float, help="Override hop-count transparency strength")
    parser.add_argument("--beta", type=float, help="Override path-diversity transparency strength")
    parser.add_argument("--skeptic-fraction", type=float, help="Fraction of agents initialized as skeptics")
    parser.add_argument("--skeptic-threshold-bias", type=float, help="Extra skepticism added to skeptic agents")
    parser.add_argument("--weak-believer-influence", type=float, help="Influence weight for weak believers")
    parser.add_argument("--weakening-per-step-min", type=int, help="Minimum number of strong believers weakened each step")
    parser.add_argument("--weakening-per-step-max", type=int, help="Maximum number of strong believers weakened each step")
    parser.add_argument("--weak-believer-fraction-cap", type=float, help="Maximum fraction of believers allowed to be weak")
    parser.add_argument("--threshold-mean", type=float, help="Override agent threshold mean")
    parser.add_argument("--threshold-std", type=float, help="Override agent threshold std")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.quick:
        config = apply_quick_mode(config)
    config = apply_cli_overrides(config, args)

    label = "quick" if args.quick else "full"
    run_pipeline(config, label)
