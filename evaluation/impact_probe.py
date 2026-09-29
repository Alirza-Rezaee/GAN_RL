"""Measure what a single market order of size q actually costs, and how far it
moves the price, using paired "twin days".

The question
------------
Stage 0 found the environment flat: four different execution schedules all tied
with TWAP. The diagnosis was that 100-share child orders do not move a ~$1000
stock, so the entire cost being optimised is one tick of spread, paid equally by
every schedule. This script measures that directly, so a new order size can be
chosen on evidence rather than guessed.

Method
------
For each seed, the same market day is run twice:

    baseline : HOLD on every step
    probe    : one MKT sell of size q on the first step, then HOLD

Because seeding immediately before reset() replays a day exactly, the paired
difference in mid price at each step is caused only by our order.

Why that is valid: the kernel's own RNG is consumed in exactly one place,
kernel.py:589, inside the branch taken only when agent_latency_model is None.
rmsc04 sets a latency model and core_environment forwards it, so sending one
extra order cannot shift any random stream. Background agents hold their own
RandomState, so any divergence between the arms is causal.

Two measurements, and only one of them is trusted
-------------------------------------------------
1. EXECUTION COST - what the probe order itself paid against the arrival mid.
   Computed from the order's own fills. Needs no baseline arm, so it carries none
   of the twin-day divergence noise, and it is exactly what Implementation
   Shortfall charges. This is the decisive number.

2. PAIRED MID IMPACT - the twin-day difference at each step. Reported, but never
   decisive: the pilot showed the late-step difference flips sign across seeds
   and barely scales with q, because background agents amplify any perturbation
   chaotically. The book also refills within the 60-second wakeup interval, so a
   minute-granularity mid difference misses most of the impact.

The baseline arm is shared across order sizes
---------------------------------------------
It sends no orders, so parent_order_size and order_fixed_size affect only
normalisation, never the simulation. That cuts the run from 800 episodes to 500.
The assumption is re-verified at runtime on the first few seeds rather than
trusted - see --verify-baseline-seeds.

Example
-------
    python -m evaluation.impact_probe --probe-config configs/impact_probe.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import abides_markets.agents.utils as markets_agent_utils

from common.env_factory import PROJECT_ROOT, load_config, make_env
from common.provenance import write_config_used
from common.seeds import assert_seeds_allowed, seed_band_name


# Row of the observation holding best_ask - best_bid, confirmed against
# markets_execution_environment_v0.raw_state_to_state and verified numerically
# against info["spread"].
SPREAD_ROW = 6

ACTION_MARKET = 0
ACTION_HOLD = 2

DEPTH_LEVELS = 5


def build_config(base_config, q, multiplier):
    """Base gate config with the probe's order size substituted in.

    parent_order_size is q * multiplier so that one child order can never finish
    the parent order; the episode then runs for the whole execution window and we
    observe the price path after the probe.
    """
    config = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in base_config.items()
    }
    config["environment"]["order_fixed_size"] = q
    config["environment"]["parent_order_size"] = q * multiplier
    return config


def read_mid(env):
    """Mid price from the latest order-book snapshot the agent holds.

    Read straight from the book rather than inverted out of the observation's
    price_impact feature: same number, without a log/exp round trip.
    """
    market_data = env.unwrapped.gym_agent.parsed_mkt_data
    return markets_agent_utils.get_mid_price(
        market_data["bids"],
        market_data["asks"],
        market_data["last_transaction"],
    )


def read_bid_depth(env, levels=DEPTH_LEVELS):
    """Shares resting at each of the top `levels` bid prices."""
    bids = env.unwrapped.gym_agent.parsed_mkt_data["bids"]
    return [bids[i][1] if i < len(bids) else 0 for i in range(levels)]


def run_arm(env, seed, total_steps, probe_size):
    """
    Run one episode. If probe_size is None the agent holds every step (baseline);
    otherwise it sends one market order on the first step and then holds.

    Returns the per-step mid prices plus, for the probe arm, everything needed to
    price the order it sent.
    """
    env.seed(seed)
    observation = env.reset()

    entry_price = env.unwrapped.entry_price
    spread_at_arrival = float(observation[SPREAD_ROW, 0])
    bid_depth = read_bid_depth(env)

    mids = []
    slippage_reward_total = 0.0
    step = 0
    info = {}
    done = False

    while not done:
        action = ACTION_MARKET if (step == 0 and probe_size is not None) else ACTION_HOLD
        observation, _, done, info = env.step(action)
        slippage_reward_total += info["slippage_reward"]
        mids.append(read_mid(env))
        step += 1

        if step > total_steps:
            raise RuntimeError(
                f"Episode did not end after {total_steps} steps (seed {seed})."
            )

    result = {
        "entry_price": entry_price,
        "spread_at_arrival": spread_at_arrival,
        "bid_depth": bid_depth,
        "mids": mids,
        "steps": step,
    }

    if probe_size is None:
        return result

    # The probe sends exactly one order, so every fill recorded this episode
    # belongs to it. parsed_episode_executed_orders holds (quantity, price) and,
    # unlike the inter-wakeup list, is not cleared by new_step_reset().
    fills = list(env.unwrapped.gym_agent.parsed_episode_executed_orders)
    filled = sum(quantity for quantity, _ in fills)

    if filled > 0:
        avg_fill_price = sum(q * p for q, p in fills) / filled
        # SELL: a fill below the arrival mid is a cost, so flip the sign.
        cost_from_fills = (entry_price - avg_fill_price) / entry_price * 1e4
        # Independent route to the same number, through the environment's own
        # accounting: slippage_reward = signed_pnl / parent_order_size.
        signed_pnl = slippage_reward_total * env.unwrapped.parent_order_size
        cost_from_reward = -signed_pnl / (entry_price * filled) * 1e4
    else:
        avg_fill_price = None
        cost_from_fills = None
        cost_from_reward = None

    result.update(
        {
            "fill_quantity": filled,
            "fill_rate": filled / probe_size,
            "avg_fill_price": avg_fill_price,
            "cost_bps_from_fills": cost_from_fills,
            "cost_bps_from_reward": cost_from_reward,
            "n_fills": len(fills),
        }
    )
    return result


def verify_baseline_invariance(base_config, multiplier, order_sizes, seeds, total_steps):
    """Re-prove that the baseline arm does not depend on q, instead of assuming it.

    The baseline is run once per seed and reused for every order size, which is
    only sound if parent_order_size and order_fixed_size cannot reach the
    simulation. They should not - the baseline sends no orders - but a silent
    failure here would corrupt every paired impact number in the output.
    """
    if len(order_sizes) < 2 or not seeds:
        return

    first, last = order_sizes[0], order_sizes[-1]
    env_a = make_env(build_config(base_config, first, multiplier))
    env_b = make_env(build_config(base_config, last, multiplier))
    try:
        for seed in seeds:
            a = run_arm(env_a, seed, total_steps, None)
            b = run_arm(env_b, seed, total_steps, None)
            if a["entry_price"] != b["entry_price"] or a["mids"] != b["mids"]:
                raise RuntimeError(
                    f"Baseline arm is NOT invariant to order size on seed {seed} "
                    f"(q={first} vs q={last}). The shared-baseline optimisation is "
                    f"invalid; every paired impact number would be wrong."
                )
    finally:
        env_a.close()
        env_b.close()

    print(
        f"  baseline invariance re-verified on {len(seeds)} seed(s): "
        f"q={first} and q={last} give identical mid paths",
        flush=True,
    )


def total_decision_steps(config):
    from abides_core.utils import str_to_ns

    environment_config = config["environment"]
    return int(
        str_to_ns(environment_config["execution_window"])
        / str_to_ns(environment_config["timestep_duration"])
    )


def summarise(episodes, criteria):
    """One row per order size, plus the pass/fail the criteria file pre-registered."""
    min_cost = criteria["primary"]["min_median_cost_bps"]
    min_fill = criteria["primary"]["min_median_fill_rate"]

    step_columns = [c for c in episodes.columns if c.startswith("mid_diff_step_")]

    rows = []
    for q, group in episodes.groupby("q", sort=True):
        costs = group["cost_bps_from_fills"].dropna()
        row = {
            "q": int(q),
            "episodes": len(group),
            "fill_rate_median": float(group["fill_rate"].median()),
            "fill_rate_mean": float(group["fill_rate"].mean()),
            "cost_median_bps": float(costs.median()) if len(costs) else None,
            "cost_mean_bps": float(costs.mean()) if len(costs) else None,
            "cost_iqr_bps": (
                float(costs.quantile(0.75) - costs.quantile(0.25)) if len(costs) else None
            ),
            "spread_at_arrival_median": float(group["spread_at_arrival"].median()),
            "bid_depth_l1_5_median": float(group["bid_depth_l1_5_total"].median()),
            "n_fills_median": float(group["n_fills"].median()),
        }
        for column in step_columns:
            values = group[column].dropna()
            row[f"{column}_median"] = float(values.median()) if len(values) else None
            row[f"{column}_iqr"] = (
                float(values.quantile(0.75) - values.quantile(0.25)) if len(values) else None
            )

        row["meets_cost_bar"] = bool(
            row["cost_median_bps"] is not None and row["cost_median_bps"] >= min_cost
        )
        row["meets_fill_bar"] = bool(row["fill_rate_median"] >= min_fill)
        row["material"] = bool(row["meets_cost_bar"] and row["meets_fill_bar"])
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("q").reset_index(drop=True)

    chosen = summary[summary["material"]]
    summary.attrs["selected_q"] = int(chosen.iloc[0]["q"]) if len(chosen) else None
    return summary


def print_report(summary, criteria):
    min_cost = criteria["primary"]["min_median_cost_bps"]
    min_fill = criteria["primary"]["min_median_fill_rate"]

    print(
        f"\nPrimary criterion: smallest q with median cost >= {min_cost} bps "
        f"AND median fill rate >= {min_fill:.0%}"
    )
    header = (
        f"{'q':>6} {'fill%':>7} {'cost med':>9} {'cost mean':>10} {'cost IQR':>9} "
        f"{'depth L1-5':>11} {'fills':>6}  verdict"
    )
    print(header)
    print("-" * len(header))
    for row in summary.itertuples():
        marks = []
        if not row.meets_cost_bar:
            marks.append("cost too small")
        if not row.meets_fill_bar:
            marks.append("fill too low")
        verdict = "MATERIAL" if row.material else " / ".join(marks)
        print(
            f"{row.q:>6} {100 * row.fill_rate_median:>7.1f} "
            f"{row.cost_median_bps:>9.3f} {row.cost_mean_bps:>10.3f} "
            f"{row.cost_iqr_bps:>9.3f} {row.bid_depth_l1_5_median:>11.0f} "
            f"{row.n_fills_median:>6.0f}  {verdict}"
        )

    selected = summary.attrs.get("selected_q")
    if selected is None:
        print(
            "\nNo tested order size makes impact material. The size grid must be "
            "widened before any new gate is designed."
        )
    else:
        print(f"\nSelected order size: q = {selected}")

    print("\nPaired twin-day mid impact, median bps of arrival price (secondary,")
    print("not decisive - see criteria.secondary.why_not_decisive):")
    step_medians = [c for c in summary.columns if c.endswith("_median") and "mid_diff_step_" in c]
    print(f"{'q':>6} " + " ".join(f"{c.split('_')[3]:>8}" for c in step_medians))
    for row in summary.itertuples(index=False):
        values = " ".join(f"{getattr(row, c):>8.3f}" for c in step_medians)
        print(f"{int(row.q):>6} {values}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure execution cost and price impact against order size."
    )
    parser.add_argument(
        "--probe-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "impact_probe.yaml",
        help="Probe design and pre-registered pass criteria.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "impact_probe_summary.csv",
    )
    parser.add_argument(
        "--episodes-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "impact_probe_episodes.csv",
    )
    parser.add_argument(
        "--verify-baseline-seeds",
        type=int,
        default=3,
        help="How many seeds to re-check the shared-baseline assumption on.",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Allow the reserved final-test seeds from configs/seed_registry.yaml.",
    )
    args = parser.parse_args()

    with args.probe_config.open("r", encoding="utf-8") as handle:
        probe_config = yaml.safe_load(handle)

    probe = probe_config["probe"]
    criteria = probe_config["criteria"]
    order_sizes = list(probe["order_sizes"])
    seed_start = probe["seeds"]["start"]
    seed_end = probe["seeds"]["end"]
    seeds = list(range(seed_start, seed_end + 1))
    multiplier = probe["parent_order_size_multiplier"]

    assert_seeds_allowed(seeds, args.final_test, context="impact_probe")

    base_config = load_config(PROJECT_ROOT / probe["base_config"])
    if base_config["environment"].get("env_variant") != probe["env_variant"]:
        parser.error(
            f"{probe['base_config']} has env_variant "
            f"{base_config['environment'].get('env_variant')!r}, but the probe "
            f"config requires {probe['env_variant']!r}."
        )
    total_steps = total_decision_steps(base_config)

    for q in order_sizes:
        if (q * multiplier) % q != 0:
            parser.error(f"parent_order_size must be a multiple of q for q={q}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = write_config_used(
        args.out.with_name(args.out.stem + "_config_used.yaml"),
        "evaluation/impact_probe.py",
        {
            "seeds": {
                "start": seed_start,
                "end": seed_end,
                "episodes": len(seeds),
                "band": seed_band_name(seed_start, seed_end),
            },
            "final_test": args.final_test,
            "probe_config_file": str(args.probe_config),
            "probe_config": probe_config,
            "base_environment": base_config["environment"],
            "decision_steps_per_episode": total_steps,
        },
    )

    episode_total = len(seeds) * (1 + len(order_sizes))
    print(
        f"{len(order_sizes)} order sizes x {len(seeds)} seeds, plus one shared "
        f"baseline per seed = {episode_total} simulated market days.",
        flush=True,
    )

    print("Verifying the shared-baseline assumption before relying on it...", flush=True)
    verify_baseline_invariance(
        base_config,
        multiplier,
        order_sizes,
        seeds[: args.verify_baseline_seeds],
        total_steps,
    )

    # One baseline per seed, reused by every order size.
    print(f"\nBaseline arm: {len(seeds)} episodes ...", end="", flush=True)
    baseline_env = make_env(build_config(base_config, order_sizes[0], multiplier))
    baselines = {}
    try:
        for seed in seeds:
            baselines[seed] = run_arm(baseline_env, seed, total_steps, None)
    finally:
        baseline_env.close()
    print(" done", flush=True)

    records = []
    for q in order_sizes:
        print(f"probe q={q}: {len(seeds)} episodes ...", end="", flush=True)
        env = make_env(build_config(base_config, q, multiplier))
        try:
            for seed in seeds:
                probe_result = run_arm(env, seed, total_steps, q)
                baseline = baselines[seed]

                if probe_result["entry_price"] != baseline["entry_price"]:
                    raise RuntimeError(
                        f"Arrival price differs between arms on seed {seed} "
                        f"(q={q}). The twin-day pairing is broken."
                    )

                record = {
                    "q": q,
                    "seed": seed,
                    "fill_quantity": probe_result["fill_quantity"],
                    "fill_rate": probe_result["fill_rate"],
                    "avg_fill_price": probe_result["avg_fill_price"],
                    "entry_price": probe_result["entry_price"],
                    "cost_bps_from_fills": probe_result["cost_bps_from_fills"],
                    "cost_bps_from_reward": probe_result["cost_bps_from_reward"],
                    "n_fills": probe_result["n_fills"],
                    "spread_at_arrival": probe_result["spread_at_arrival"],
                    "bid_depth_l1_5_total": sum(probe_result["bid_depth"]),
                }
                for level, shares in enumerate(probe_result["bid_depth"], start=1):
                    record[f"bid_depth_l{level}"] = shares

                for index in range(total_steps):
                    probe_mid = (
                        probe_result["mids"][index]
                        if index < len(probe_result["mids"]) else None
                    )
                    base_mid = (
                        baseline["mids"][index] if index < len(baseline["mids"]) else None
                    )
                    # Recorded in bps of the arrival price, so days at different
                    # price levels are comparable. Multiply by entry_price/1e4 to
                    # recover price units.
                    record[f"mid_diff_step_{index}"] = (
                        (probe_mid - base_mid) / probe_result["entry_price"] * 1e4
                        if probe_mid is not None and base_mid is not None else None
                    )

                records.append(record)
        finally:
            env.close()

        done = [r for r in records if r["q"] == q]
        costs = [r["cost_bps_from_fills"] for r in done if r["cost_bps_from_fills"] is not None]
        print(
            f" cost median={np.median(costs):.3f} bps, "
            f"fill median={np.median([r['fill_rate'] for r in done]):.1%}",
            flush=True,
        )

    episodes = pd.DataFrame(records)

    # The two cost routes are independent; if they disagree, one of them is wrong.
    both = episodes.dropna(subset=["cost_bps_from_fills", "cost_bps_from_reward"])
    max_gap = float((both["cost_bps_from_fills"] - both["cost_bps_from_reward"]).abs().max())
    print(f"\nCost cross-check: max |fills route - reward route| = {max_gap:.2e} bps")

    summary = summarise(episodes, criteria)

    args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.episodes_out, index=False)
    summary.to_csv(args.out, index=False)

    print_report(summary, criteria)
    print(f"\nSaved summary to:     {args.out}")
    print(f"Saved per-episode to: {args.episodes_out}")
    print(f"Saved provenance to:  {provenance_path}")


if __name__ == "__main__":
    main()
