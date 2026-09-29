"""Run the hand-written execution rules on the gate configs and test them
against TWAP on the same market days.

Purpose: "headroom screening"
-----------------------------
Before spending days training PPO, answer a much cheaper question: does this
environment reward ANY non-TWAP schedule at all? Four different hand-made
schedules are run against TWAP. If all four tie with TWAP, the environment is
flat and an RL agent would have nothing to discover either. No RL is involved
here, and nothing in this file learns anything.

gate_b has zero slack (parent_order_size == order_fixed_size * total_steps), so
the safety net in ManualRule.predict() forces a market order on every step and
every rule must come out bit-identical to TWAP. gate_d halves the parent order,
which creates five steps of slack, so there the rules can actually express an
opinion. gate_b is the control; gate_d is the test.

Everything the environment exposes was read from the real source, not guessed
--------------------------------------------------------------------------
Action mapping - abides/abides-gym/abides_gym/envs/markets_execution_environment_v0.py,
    _map_action_space_to_ABIDES_SIMULATOR_SPACE(): 0 -> MKT, 1 -> LMT priced at
    near_touch, 2 -> DO NOTHING. See evaluation/manual_rules.py for the verbatim
    quote.

Spread - row 6 of the (11, 1) observation. raw_state_to_state() builds
    [holdings_pct, time_pct, diff_pct, imbalance_all, imbalance_5, price_impact,
    spread, direction_feature] + 3 padded returns, and num_state_features is
    8 + state_history_length - 1 = 11. Checked numerically as well:
    observation[6, 0] equals info["spread"] on every step.

Info keys - raw_state_to_info() returns asdict(self.custom_metrics_tracker)
    whenever debug_mode is False, which is the default and which none of our
    configs override. So info["remaining_quantity"], info["executed_quantity"]
    and info["slippage_reward"] all exist. Their docstrings say "at the end of
    the episode", but core_environment.step() calls raw_state_to_done() (which
    writes them) BEFORE raw_state_to_info(), so they are refreshed every step.

Implementation shortfall - NOT in info. It is computed here through
    common.metrics.implementation_shortfall_bps(), the same function
    evaluate_twap_gates.py uses, because otherwise the two CSVs would not be
    comparable.

Why the episode loop is copied from the TWAP evaluator
-----------------------------------------------------
run_rule_episode() below deliberately mirrors evaluate_twap_gates.run_twap_episode():
one environment per (rule, gate), then env.seed(seed) + env.reset() per episode.
core_environment.reset() draws the ABIDES seed from self.np_random, so seeding
the env immediately before reset makes seed N replay the identical simulated
market day for every policy. That identity is the whole basis of the paired
test. It is also verified: on gate_b seed 100 all four rules return
is_bps = 1.8485268301465916, and the TWAP baseline file records 1.848527 for
gate_b seed 100.

Example
-------
    python -m evaluation.evaluate_manual_rules_gates \
        --configs configs/gate_b.yaml configs/gate_d.yaml \
        --seed-start 100 --seed-end 299 \
        --twap-episodes-csv results/twap_gate_bvd_episodes.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

from abides_core.utils import str_to_ns

from common.env_factory import PROJECT_ROOT, load_config, make_env
from common.metrics import implementation_shortfall_bps
from common.provenance import write_config_used as write_provenance
from common.seeds import assert_seeds_allowed, seed_band_name
from evaluation.manual_rules import (
    ACTION_HOLD,
    ACTION_LIMIT,
    ACTION_MARKET,
    ALL_RULES,
    RuleState,
    make_rule,
)


# Row of the observation holding best_ask - best_bid. See the module docstring.
SPREAD_ROW = 6

ACTION_LABELS = {ACTION_MARKET: "mkt", ACTION_LIMIT: "lmt", ACTION_HOLD: "hold"}

DEFAULT_TWAP_EPISODES_CSV = PROJECT_ROOT / "results" / "twap_gate_bvd_episodes.csv"
DEFAULT_GATE_THRESHOLDS = PROJECT_ROOT / "configs" / "gate.yaml"

# Paired differences smaller than this are treated as the exact zero they really
# are. See paired_test_vs_twap() for why this is not cosmetic.
IDENTICAL_TOLERANCE_BPS = 1e-9


def total_decision_steps(config):
    """How many times the agent is asked for an action inside one episode.

    Identical formula to baselines/twap_agent.py: the agent is woken once every
    timestep_duration and must finish the parent order within execution_window.
    Derived from the clock, NOT from parent_order_size / order_fixed_size -
    those two are independent, and that independence is exactly what lets gate_d
    create slack.
    """
    environment_config = config["environment"]
    execution_window_ns = str_to_ns(environment_config["execution_window"])
    timestep_ns = str_to_ns(environment_config["timestep_duration"])
    return int(execution_window_ns / timestep_ns)


def check_gate_config(config, gate_name):
    """Refuse to run on a config that cannot produce a meaningful number.

    parent_order_size must be an exact multiple of order_fixed_size. Otherwise
    the final child order overshoots, the episode collects the
    too_much_reward_update penalty, and that penalty has nothing to do with the
    policy under test.
    """
    environment_config = config["environment"]
    parent_order_size = environment_config["parent_order_size"]
    order_fixed_size = environment_config["order_fixed_size"]
    steps = total_decision_steps(config)

    if parent_order_size % order_fixed_size != 0:
        raise ValueError(
            f"{gate_name}: parent_order_size={parent_order_size} is not an exact "
            f"multiple of order_fixed_size={order_fixed_size}. The last child "
            f"order would oversell."
        )

    orders_needed = parent_order_size // order_fixed_size
    if orders_needed > steps:
        raise ValueError(
            f"{gate_name}: {orders_needed} child orders are needed but the "
            f"episode only has {steps} decision steps. The parent order could "
            f"never complete, whatever the policy does."
        )

    return parent_order_size, order_fixed_size, steps


def run_rule_episode(env, rule, parent_order_size, order_fixed_size, total_steps, seed):
    """Run one reproducible episode driven by `rule` and return its metrics."""
    rule.reset()
    env.seed(seed)
    observation = env.reset()

    total_slippage_reward = 0.0
    spread_history = []
    remaining_shares = parent_order_size
    # Counted locally on purpose. info["action_counter"] lives on the
    # CustomMetricsTracker, which is built once in the environment's __init__ and
    # never reset, so on a reused env it accumulates over every episode.
    action_counts = {label: 0 for label in ACTION_LABELS.values()}
    step = 0
    info = {}

    while True:
        current_spread = float(observation[SPREAD_ROW, 0])

        state = RuleState(
            step=step,
            total_steps=total_steps,
            remaining_shares=remaining_shares,
            order_fixed_size=order_fixed_size,
            spread=current_spread,
            spread_history=list(spread_history),
        )
        action = rule.predict(state)
        action_counts[ACTION_LABELS[action]] += 1

        observation, _, done, info = env.step(action)

        total_slippage_reward += info["slippage_reward"]
        remaining_shares = info["remaining_quantity"]
        spread_history.append(current_spread)
        step += 1

        if done:
            break

        if step >= total_steps:
            # The environment sets done once current_time reaches
            # mkt_open + first_interval + execution_window, which is exactly
            # total_steps wake-ups. Reaching this line means the clock and the
            # config disagree, and every metric below would be wrong.
            raise RuntimeError(
                f"Episode did not terminate after {total_steps} steps "
                f"(seed {seed}). The step budget derived from the config and "
                f"the environment's own time limit disagree."
            )

    shortfall_bps = implementation_shortfall_bps(
        total_slippage_reward=total_slippage_reward,
        entry_price=env.unwrapped.entry_price,
        executed_quantity=info["executed_quantity"],
        parent_order_size=parent_order_size,
    )

    episode = {
        "seed": seed,
        "is_bps": shortfall_bps,
        "remaining_quantity": remaining_shares,
        "completed": remaining_shares == 0,
        "oversold": remaining_shares < 0,
        "steps_used": step,
        # Present only on the variable_last_child variant (common/execution_env.py);
        # None on the stock environment. Needed to explain any residual oversell.
        "clamped_orders": getattr(env.unwrapped, "clamped_order_count", None),
        "suppressed_orders": getattr(env.unwrapped, "suppressed_order_count", None),
    }
    episode.update({f"n_{label}": count for label, count in action_counts.items()})
    return episode


def run_rule_on_gate(rule_name, config, gate_name, seed_start, seed_end):
    """Run one rule over the whole seed range on one gate."""
    parent_order_size, order_fixed_size, steps = check_gate_config(config, gate_name)

    env = make_env(config)
    episodes = []
    try:
        for seed in range(seed_start, seed_end + 1):
            # A fresh rule object per episode, so no rule can leak state from
            # one market day into the next.
            rule = make_rule(rule_name)
            episodes.append(
                run_rule_episode(
                    env, rule, parent_order_size, order_fixed_size, steps, seed
                )
            )
    finally:
        env.close()

    frame = pd.DataFrame(episodes)
    frame.insert(0, "rule", rule_name)
    frame.insert(1, "gate", gate_name)
    return frame


def load_twap_episodes(path):
    """Load the PER-EPISODE TWAP results, which is what pairing needs.

    The TWAP summary CSV has one row per gate and no seed column, so it cannot
    be used here. evaluate_twap_gates.py writes the per-episode file through its
    --episodes-out flag.
    """
    # float_precision="round_trip" matters more than it looks. pandas' default
    # float parser is not bit-exact: this file stores gate_b seed 100 as
    # 1.8485268301465916 and the default parser hands back 1.848526830146592, a
    # difference of 4.4e-16. On gate_b every rule is literally the same policy as
    # TWAP, so every paired difference must be exactly zero; a 1e-16 parsing
    # error turns 200 true zeros into 200 tiny signed differences that
    # wilcoxon() would rank into a meaningless p-value.
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {"gate", "seed", "is_bps"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(
            f"{path} is missing column(s) {sorted(missing)}. This must be the "
            f"per-episode TWAP output (--episodes-out of "
            f"evaluation/evaluate_twap_gates.py), not the summary CSV."
        )
    return frame[["gate", "seed", "is_bps"]].rename(columns={"is_bps": "twap_is_bps"})


def holm_adjust(p_values):
    """Holm-Bonferroni correction.

    One test is run per rule inside each gate, so with four rules the chance
    that at least one looks significant by luck alone is about 19%, not 5%.
    Holm controls that while being less blunt than plain Bonferroni: the
    smallest p is multiplied by m, the next by m-1, and so on.

    Entries that are None (no test was possible, e.g. every paired difference
    was exactly zero) stay None and are excluded from the count m.
    """
    testable = [(index, p) for index, p in enumerate(p_values) if p is not None]
    adjusted = [None] * len(p_values)

    number_of_tests = len(testable)
    running_maximum = 0.0
    for rank, (index, p_value) in enumerate(sorted(testable, key=lambda pair: pair[1])):
        candidate = min(1.0, (number_of_tests - rank) * p_value)
        # Holm's adjusted p-values must be non-decreasing along the sorted order.
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum

    return adjusted


def paired_test_vs_twap(group):
    """Paired Wilcoxon signed-rank of one rule against TWAP on one gate.

    IS is a COST, so `difference = rule - twap` is negative when the rule is
    cheaper, i.e. better. improvement_bps flips the sign so that positive means
    "the rule saved this many bps", which is easier to read.
    """
    usable = group.dropna(subset=["is_bps", "twap_is_bps"])
    rule_is = usable["is_bps"].to_numpy(dtype=float)
    twap_is = usable["twap_is_bps"].to_numpy(dtype=float)
    differences = rule_is - twap_is
    # Second line of defence behind float_precision="round_trip". IS values here
    # are of order 1 bps and the pre-registered minimum effect is 0.10 bps, so
    # snapping anything below 1e-9 bps cannot hide a real result - it only stops
    # double-precision dust from being ranked as if it were signal.
    differences[np.abs(differences) < IDENTICAL_TOLERANCE_BPS] = 0.0

    result = {
        "paired_n": int(len(differences)),
        "n_identical_to_twap": int(np.sum(differences == 0)),
        "mean_improvement_bps": None,
        "median_improvement_bps": None,
        "wilcoxon_stat": None,
        "wilcoxon_p": None,
    }

    if len(differences) == 0:
        return result

    result["mean_improvement_bps"] = float(-np.mean(differences))
    result["median_improvement_bps"] = float(-np.median(differences))

    if np.any(differences != 0):
        # Test the snapped differences, not (rule_is, twap_is). wilcoxon(x, y) is
        # defined as wilcoxon(x - y), so passing the two arrays would silently
        # undo the tolerance applied above.
        statistic, p_value = wilcoxon(differences, alternative="two-sided")
        result["wilcoxon_stat"] = float(statistic)
        result["wilcoxon_p"] = float(p_value)
    # else: every difference is exactly zero. wilcoxon() cannot run on that, and
    # reporting it as "no difference found" would be misleading - the rule is
    # not merely statistically indistinguishable from TWAP, it is the same
    # policy. n_identical_to_twap carries that.

    return result


def summarise(rule_episodes, twap_episodes, thresholds):
    """Build the (rule x gate) table, apply Holm within each gate, and decide."""
    merged = rule_episodes.merge(twap_episodes, on=["gate", "seed"], how="left")

    missing_baseline = merged["twap_is_bps"].isna() & merged["is_bps"].notna()
    if missing_baseline.any():
        absent = merged.loc[missing_baseline, ["gate", "seed"]].drop_duplicates()
        print(
            f"  WARNING: no TWAP baseline for {len(absent)} (gate, seed) pairs; "
            f"they are excluded from the paired test.",
            flush=True,
        )

    screening = thresholds["screening"]
    alpha = screening["test"]["alpha"]
    min_improvement = screening["effect"]["min_median_improvement_bps"]
    min_completion = screening["validity"]["min_completion_pct"]
    max_oversell = screening["validity"]["max_oversell_episodes"]

    rows = []
    for (gate, rule), group in merged.groupby(["gate", "rule"], sort=False):
        executed = group["is_bps"].dropna()
        row = {
            "gate": gate,
            "rule": rule,
            "episodes": int(len(group)),
            "is_mean_bps": float(executed.mean()) if len(executed) else None,
            "is_median_bps": float(executed.median()) if len(executed) else None,
            "completed_pct": float(group["completed"].mean() * 100),
            "mean_unexecuted_shares": float(group["remaining_quantity"].clip(lower=0).mean()),
            "oversell_episodes": int(group["oversold"].sum()),
            "episodes_without_fills": int(group["is_bps"].isna().sum()),
            "mean_steps_used": float(group["steps_used"].mean()),
            "n_mkt": int(group["n_mkt"].sum()),
            "n_lmt": int(group["n_lmt"].sum()),
            "n_hold": int(group["n_hold"].sum()),
            "clamped_orders": (
                int(group["clamped_orders"].sum())
                if group["clamped_orders"].notna().all() else None
            ),
            "suppressed_orders": (
                int(group["suppressed_orders"].sum())
                if group["suppressed_orders"].notna().all() else None
            ),
        }
        row.update(paired_test_vs_twap(group))
        rows.append(row)

    summary = pd.DataFrame(rows)

    # Holm is applied across the rules WITHIN each gate, per configs/gate.yaml.
    summary["holm_p"] = None
    for gate in summary["gate"].unique():
        mask = summary["gate"] == gate
        raw = [None if pd.isna(p) else float(p) for p in summary.loc[mask, "wilcoxon_p"]]
        summary.loc[mask, "holm_p"] = pd.Series(
            holm_adjust(raw), index=summary.loc[mask].index, dtype=object
        )

    def verdict(row):
        if row["oversell_episodes"] > max_oversell:
            return "fail_oversell"
        if row["completed_pct"] < min_completion:
            return "fail_completion"
        if row["n_identical_to_twap"] == row["paired_n"] and row["paired_n"] > 0:
            return "identical_to_twap"
        if row["holm_p"] is None or pd.isna(row["holm_p"]):
            return "not_testable"
        if row["holm_p"] >= alpha:
            return "not_significant"
        if row["median_improvement_bps"] is None:
            return "not_testable"
        if row["median_improvement_bps"] < min_improvement:
            if row["median_improvement_bps"] <= -min_improvement:
                return "significantly_worse"
            return "significant_but_below_min_effect"
        return "PASS"

    summary["verdict"] = summary.apply(verdict, axis=1)
    return summary


def write_config_used(path, args, gate_configs, thresholds):
    """Project rule 6: every run records its commit, config, seeds and date."""
    payload = {
        "seeds": {
            "start": args.seed_start,
            "end": args.seed_end,
            "episodes_per_rule_per_gate": args.seed_end - args.seed_start + 1,
            "band": seed_band_name(args.seed_start, args.seed_end),
        },
        "final_test": args.final_test,
        "rules": list(args.rules),
        "twap_baseline_csv": str(args.twap_episodes_csv),
        "gate_thresholds_file": str(args.gate_thresholds),
        "gate_thresholds": thresholds,
        "configs": {
            gate_name: {
                "path": str(config_path),
                "decision_steps_per_episode": total_decision_steps(config),
                "environment": config["environment"],
            }
            for gate_name, (config_path, config) in gate_configs.items()
        },
    }
    return write_provenance(
        path, "evaluation/evaluate_manual_rules_gates.py", payload
    )


def print_table(summary):
    """Console view of the columns that drive the decision."""
    header = (
        f"{'gate':8s} {'rule':20s} {'IS mean':>9s} {'IS med':>8s} "
        f"{'impr med':>9s} {'compl%':>7s} {'unexec':>7s} {'over':>5s} "
        f"{'raw p':>10s} {'holm p':>10s}  verdict"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summary.itertuples():
        def number(value, width, digits=3):
            if value is None or pd.isna(value):
                return f"{'n/a':>{width}s}"
            return f"{value:>{width}.{digits}f}"

        print(
            f"{row.gate:8s} {row.rule:20s} "
            f"{number(row.is_mean_bps, 9)} {number(row.is_median_bps, 8)} "
            f"{number(row.median_improvement_bps, 9)} "
            f"{row.completed_pct:>7.2f} {row.mean_unexecuted_shares:>7.2f} "
            f"{row.oversell_episodes:>5d} "
            f"{number(row.wilcoxon_p, 10, 4)} {number(row.holm_p, 10, 4)}  "
            f"{row.verdict}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the manual execution rules on the gate configs and "
                    "test each one against TWAP on the same seeds."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=[PROJECT_ROOT / "configs" / "gate_b.yaml",
                 PROJECT_ROOT / "configs" / "gate_d.yaml"],
        help="Gate YAML files to evaluate.",
    )
    parser.add_argument(
        "--rules",
        nargs="+",
        default=list(ALL_RULES.keys()),
        choices=list(ALL_RULES.keys()),
        help="Which rules to run (default: all four).",
    )
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-end", type=int, default=299)
    parser.add_argument(
        "--twap-episodes-csv",
        type=Path,
        default=DEFAULT_TWAP_EPISODES_CSV,
        help="Per-episode TWAP results, used as the paired baseline.",
    )
    parser.add_argument(
        "--gate-thresholds",
        type=Path,
        default=DEFAULT_GATE_THRESHOLDS,
        help="Decision thresholds, fixed before the results were seen.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "manual_rules_gate_results.csv",
        help="Summary CSV path, one row per (rule, gate).",
    )
    parser.add_argument(
        "--episodes-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "manual_rules_gate_episodes.csv",
        help="Per-episode CSV path, so any test can be recomputed without "
             "re-running the simulation.",
    )
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Allow the reserved final-test seeds from "
             "configs/seed_registry.yaml. Only for the thesis final evaluation.",
    )
    args = parser.parse_args()

    if args.seed_end < args.seed_start:
        parser.error("--seed-end must not be smaller than --seed-start")

    assert_seeds_allowed(
        range(args.seed_start, args.seed_end + 1),
        args.final_test,
        context="evaluate_manual_rules_gates",
    )

    with args.gate_thresholds.open("r", encoding="utf-8") as handle:
        thresholds = yaml.safe_load(handle)

    configured = thresholds["screening"]["seeds"]
    if (configured["start"], configured["end"]) != (args.seed_start, args.seed_end):
        parser.error(
            f"Seed range {args.seed_start}..{args.seed_end} does not match "
            f"{args.gate_thresholds.name}, which pre-registered "
            f"{configured['start']}..{configured['end']}."
        )

    gate_configs = {}
    for config_path in args.configs:
        config = load_config(config_path)
        gate_name = Path(config_path).stem
        check_gate_config(config, gate_name)
        gate_configs[gate_name] = (config_path, config)

    twap_episodes = load_twap_episodes(args.twap_episodes_csv)

    # Written BEFORE the first episode, so the dirty-tree flag describes the tree
    # as the run started rather than as it ended.
    config_used_path = write_config_used(
        args.out.with_name(args.out.stem.replace("_results", "") + "_config_used.yaml"),
        args,
        gate_configs,
        thresholds,
    )

    episode_count = args.seed_end - args.seed_start + 1
    print(
        f"{len(args.rules)} rules x {len(gate_configs)} gates x {episode_count} "
        f"episodes = {len(args.rules) * len(gate_configs) * episode_count} "
        f"simulated market days.",
        flush=True,
    )

    frames = []
    for gate_name, (config_path, config) in gate_configs.items():
        steps = total_decision_steps(config)
        environment_config = config["environment"]
        slack = steps - environment_config["parent_order_size"] // environment_config["order_fixed_size"]
        print(
            f"\n{gate_name}: parent={environment_config['parent_order_size']}, "
            f"fixed={environment_config['order_fixed_size']}, steps={steps}, "
            f"slack={slack} step(s)",
            flush=True,
        )
        for rule_name in args.rules:
            print(f"  {rule_name} ...", end="", flush=True)
            frame = run_rule_on_gate(
                rule_name, config, gate_name, args.seed_start, args.seed_end
            )
            frames.append(frame)
            executed = frame["is_bps"].dropna()
            print(
                f" IS mean={executed.mean():.3f} median={executed.median():.3f} bps, "
                f"completed={frame['completed'].mean() * 100:.2f}%, "
                f"oversell={int(frame['oversold'].sum())}",
                flush=True,
            )

    rule_episodes = pd.concat(frames, ignore_index=True)
    summary = summarise(rule_episodes, twap_episodes, thresholds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
    rule_episodes.to_csv(args.episodes_out, index=False)
    summary.to_csv(args.out, index=False)

    print_table(summary)
    print(f"\nSaved summary to:      {args.out}")
    print(f"Saved per-episode to:  {args.episodes_out}")
    print(f"Saved provenance to:   {config_used_path}")


if __name__ == "__main__":
    main()
