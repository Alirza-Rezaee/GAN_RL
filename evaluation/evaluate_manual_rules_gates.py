"""
evaluate_manual_rules_gates.py

Runs each manual rule (from manual_rules.py) against one or more gate
configs, using the SAME seed range as your TWAP evaluation, and writes a
CSV with the same shape of columns as your TWAP results, plus a paired
Wilcoxon test against TWAP for each (rule, gate) combination.

========================================================================
READ THIS BEFORE RUNNING
========================================================================
I do not have access to your actual project files — common/env_factory.py,
evaluation/evaluate_twap_gates.py, or the real shape of the observation
and info dict your environment returns. Everything below was written to
match the DESCRIPTION of your setup from this conversation, not your real
source code.

Every spot that needs your real code is marked:

    # >>> ADAPT

Hand this file to Claude Code together with evaluate_twap_gates.py and
common/env_factory.py, and ask it — in these words or similar — to fill
in every "# >>> ADAPT" spot so it matches your actual project exactly,
and to show you a diff before saving. Do not run this file until that's
done; several of the placeholders below (env=None, made-up dict keys)
will simply crash or silently produce wrong numbers otherwise.
========================================================================
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.stats import wilcoxon

from evaluation.manual_rules import ALL_RULES, ManualRule, RuleState, make_rule
from common.env_factory import make_env
from common.metrics import implementation_shortfall_bps


def run_episode(env, rule: ManualRule, order_fixed_size: int,
                 total_steps: int, parent_order_size: int) -> Dict:
    """
    Runs one episode with `rule` making every decision. Returns a dict of
    per-episode metrics, matching (as closely as possible) the columns
    your TWAP evaluator already produces, so the two CSVs line up.
    """
    rule.reset()
    observation = env.reset()

    spread_history: List[float] = []
    remaining_shares = parent_order_size
    info: Dict = {}
    total_slippage_reward = 0.0

    for step in range(total_steps):
        # >>> ADAPT: replace with however your project actually reads the
        # spread out of `observation`. An earlier draft assumed
        # observation[6, 0] based on the published state vector
        # (holdingsPct, timePct, differencePct, imbalance5, imbalanceAll,
        # priceImpact, spread, directionFeature, R^k) — CONFIRM the exact
        # row/column against your real state-construction code before
        # trusting this index; your fork may order or buffer it differently
        # (state_history_length=4, market_data_buffer_length=5 in your
        # config suggest observation is not a flat vector).
        current_spread = float(observation[6, 0])

        state = RuleState(
            step=step,
            total_steps=total_steps,
            remaining_shares=remaining_shares,
            order_fixed_size=order_fixed_size,
            spread=current_spread,
            spread_history=list(spread_history),
        )
        action = rule.predict(state)

        observation, reward, done, info = env.step(action)
        total_slippage_reward += info["slippage_reward"]

        # >>> ADAPT: confirm the real info dict key for shares still
        # outstanding. "remaining_quantity" is a guess, not a fact.
        remaining_shares = info.get("remaining_quantity", remaining_shares)
        spread_history.append(current_spread)

        if done:
            break

    executed_quantity = info["executed_quantity"]
    is_bps = implementation_shortfall_bps(
        total_slippage_reward=total_slippage_reward,
        entry_price=env.unwrapped.entry_price,
        executed_quantity=executed_quantity,
        parent_order_size=parent_order_size,
    )

    return {
        "is_bps": is_bps,
        "completed": remaining_shares <= 0,
        "unexecuted_shares": max(remaining_shares, 0),
        "oversold": remaining_shares < 0,
    }


def load_gate_config(gate_path: str) -> dict:
    with open(gate_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def steps_from_config(config: dict) -> int:
    """
    Number of decision steps per episode. Derived from execution_window /
    timestep_duration (e.g. 10 minutes / 60s = 10), NOT from
    parent_order_size / order_fixed_size — those two are independent
    (that independence is exactly what lets gate_d create slack).

    # >>> ADAPT: replace this with however env_factory actually computes
    # it, if it isn't this simple division (e.g. it may already expose a
    # `n_steps` or `horizon` value you should just read directly).
    """
    def to_seconds(s: str) -> int:
        h, m, sec = (int(p) for p in s.split(":"))
        return h * 3600 + m * 60 + sec

    window_s = to_seconds(config["environment"]["execution_window"])
    step_s = int(config["environment"]["timestep_duration"].rstrip("s"))
    return window_s // step_s


def run_rule_on_gate(rule_name: str, gate_path: str,
                      seed_start: int, seed_end: int) -> List[Dict]:
    config = load_gate_config(gate_path)
    env_cfg = config["environment"]
    order_fixed_size = env_cfg["order_fixed_size"]
    parent_order_size = env_cfg["parent_order_size"]
    total_steps = steps_from_config(config)

    results = []
    for seed in range(seed_start, seed_end + 1):
        rule = make_rule(rule_name)
        env = make_env(config, seed=seed)
        try:
            metrics = run_episode(
                env,
                rule,
                order_fixed_size,
                total_steps,
                parent_order_size,
            )
            metrics["seed"] = seed
            results.append(metrics)
        finally:
            env.close()

    return results


def load_twap_is_by_gate_and_seed(twap_csv_path: str
                                   ) -> Dict[Tuple[str, int], float]:
    """
    Loads your existing TWAP results CSV into {(gate_name, seed): is_bps}
    so each rule's per-seed results can be paired correctly for Wilcoxon.

    # >>> ADAPT: match the real column names in your TWAP CSV. Based on
    # the CSV you pasted earlier the columns look like:
    #   gate, config, episodes, seed_start, seed_end, is_mean_bps, ...
    # — that file stores ONE ROW PER GATE (aggregated), not one row per
    # seed. If that's still true, this function needs your PER-EPISODE
    # TWAP output instead (whatever evaluate_twap_gates.py writes before
    # aggregating), not the summary CSV. Point this at that file, or add
    # a --per-seed-output flag to evaluate_twap_gates.py if it doesn't
    # already save one.
    """
    lookup: Dict[Tuple[str, int], float] = {}
    with open(twap_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gate = row["gate"]
            seed = int(row["seed"])          # >>> ADAPT if column name differs
            is_bps = float(row["is_bps"])    # >>> ADAPT if column name differs
            lookup[(gate, seed)] = is_bps
    return lookup


def paired_wilcoxon(rule_is: List[float], twap_is: List[float]
                     ) -> Tuple[Optional[float], Optional[float]]:
    if len(rule_is) != len(twap_is) or len(rule_is) < 1:
        return None, None
    try:
        stat, p = wilcoxon(rule_is, twap_is)
        return float(stat), float(p)
    except ValueError:
        # e.g. all differences are zero — wilcoxon can't run
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True,
                         help="Gate yaml paths, e.g. configs/gate_b.yaml configs/gate_d.yaml")
    parser.add_argument("--rules", nargs="+", default=list(ALL_RULES.keys()))
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=299)
    parser.add_argument("--twap-csv", required=True,
                         help="Per-seed TWAP results, used as the paired baseline.")
    parser.add_argument("--out", default="results/manual_rules_gate_results.csv")
    args = parser.parse_args()

    twap_lookup = load_twap_is_by_gate_and_seed(args.twap_csv)

    rows = []
    for gate_path in args.configs:
        gate_name = Path(gate_path).stem
        for rule_name in args.rules:
            print(f"Running {rule_name} on {gate_name} "
                  f"(seeds {args.seed_start}-{args.seed_end})...")

            rule_results = run_rule_on_gate(
                rule_name, gate_path, args.seed_start, args.seed_end
            )

            is_values = [r["is_bps"] for r in rule_results if r["is_bps"] is not None]
            twap_values = [
                twap_lookup.get((gate_name, r["seed"]))
                for r in rule_results
            ]
            # keep only seeds present on both sides, same order
            paired_rule, paired_twap = [], []
            for r, t in zip(rule_results, twap_values):
                if r["is_bps"] is not None and t is not None:
                    paired_rule.append(r["is_bps"])
                    paired_twap.append(t)

            stat, p = paired_wilcoxon(paired_rule, paired_twap)

            n = len(rule_results)
            completed_pct = 100 * sum(r["completed"] for r in rule_results) / n
            oversold_count = sum(r["oversold"] for r in rule_results)

            rows.append({
                "rule": rule_name,
                "gate": gate_name,
                "episodes": n,
                "is_mean_bps": np.mean(is_values) if is_values else None,
                "is_median_bps": np.median(is_values) if is_values else None,
                "completed_pct": completed_pct,
                "oversold_count": oversold_count,
                "paired_n": len(paired_rule),
                "wilcoxon_stat_vs_twap": stat,
                "wilcoxon_p_vs_twap": p,
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()