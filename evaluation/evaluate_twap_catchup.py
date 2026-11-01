"""
Run the two arms pre-registered in configs/twap_catchup_criteria.yaml.

What this tests
---------------
Whether gate_e's completion failure was the gate or the baseline. The per-child
fill diagnostic (45496c1) found that all 24 incomplete episodes out of 100 had a
partially filled child, that none of the 24 recovered, and that bid depth ahead
of children 1 to 5 is flat - so the earlier children are not draining the book.
baselines/twap_catchup_agent.py schedules on outstanding shares instead of sent
shares, which is the one thing that changes.

  Run A, regression: twap_catchup on gate_d_varchild, seeds 1000-1199. Its
  100-share children almost always fill, so the new agent must reproduce the
  committed TWAP arm episode for episode. Seeds 1020, 1129 and 1154 are named in
  the criteria file in advance as the only legitimate differences - they are the
  only three episodes in that arm that did not complete, so the only ones where
  a fill-aware rule can behave differently at all.

  Run B, effect: twap_catchup on gate_e, seeds 1000-1199. Completion must reach
  95%, oversell must be zero, and median IS must stay at or above 0.40 bps.

Why a separate script instead of a flag on evaluate_twap_gates.py
-----------------------------------------------------------------
Three reasons, in order of weight:

  * TWAPCatchupAgent.predict takes the outstanding quantity, not an observation.
    Teaching the existing runner both calling conventions would put a branch in
    the one function every committed TWAP number was produced by.
  * This script needs per-episode quantities evaluate_twap_gates.py does not
    record: how many child orders were actually sent, and the environment's
    clamped/suppressed counters.
  * Run A is an exact-reproduction check against a committed CSV, which is a
    different job from "compare two gates on the same seeds".

So evaluate_twap_gates.py is left exactly as it was, and every number it
produced stays reproducible from its own source.

Usage
-----
    python -m evaluation.evaluate_twap_catchup --episodes 200 --seed-start 1000
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from baselines.twap_catchup_agent import TWAPCatchupAgent
from common.env_factory import PROJECT_ROOT, load_config, make_env
from common.metrics import implementation_shortfall_bps
from common.provenance import write_config_used
from common.seeds import assert_seeds_allowed, seed_band_name

# Imported rather than redefined: a second copy of this tolerance is exactly the
# kind of thing ba3d45e consolidated the provenance writer to avoid.
from evaluation.evaluate_manual_rules_gates import IDENTICAL_TOLERANCE_BPS


# The arms, as pre-registered. Hard-coded rather than taken from the command
# line so that a run cannot quietly be pointed at a different gate than the one
# the criteria file judges.
RUN_A = {
    "name": "run_a_regression",
    "gate": "gate_d_varchild",
    "config": "configs/gate_d_varchild.yaml",
    "committed_gate": "gate_d_varchild",
}
RUN_B = {
    "name": "run_b_effect",
    "gate": "gate_e",
    "config": "configs/gate_e.yaml",
    "committed_gate": "gate_e",
}

# The committed TWAP arms both runs are paired against.
COMMITTED_EPISODES = "results/gate_e_validation_episodes.csv"

# From configs/twap_catchup_criteria.yaml, criterion
# run_a_identical_where_fills_are_complete. Restated here so the script fails
# loudly if someone edits one copy; verify_criteria_file() below checks they
# still agree.
EXPECTED_DIFFERING_SEEDS = [1020, 1129, 1154]

MIN_COMPLETED_PCT = 95.0
MAX_OVERSELL_EPISODES = 0
MIN_MEDIAN_IS_BPS = 0.40


def verify_criteria_file(path):
    """Abort unless the thresholds compiled in above still match the committed
    criteria file.

    Pre-registration is only worth anything if the run actually applies the
    registered numbers. Duplicating them in code is convenient but creates two
    places to edit, so the duplication is checked rather than trusted.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        criteria = yaml.safe_load(handle)["criteria"]

    expected = {
        "EXPECTED_DIFFERING_SEEDS": (
            criteria["run_a_identical_where_fills_are_complete"][
                "expected_differing_seeds"
            ],
            EXPECTED_DIFFERING_SEEDS,
        ),
        "MIN_COMPLETED_PCT": (
            criteria["run_b_completion"]["min_completed_pct"],
            MIN_COMPLETED_PCT,
        ),
        "MAX_OVERSELL_EPISODES": (
            criteria["run_b_oversell"]["max_oversell_episodes"],
            MAX_OVERSELL_EPISODES,
        ),
        "MIN_MEDIAN_IS_BPS": (
            criteria["run_b_cost_is_material"]["min_median_is_bps"],
            MIN_MEDIAN_IS_BPS,
        ),
    }

    disagreements = [
        f"{name}: criteria file says {in_file!r}, script says {in_code!r}"
        for name, (in_file, in_code) in expected.items()
        if in_file != in_code
    ]
    if disagreements:
        raise SystemExit(
            "Pre-registered criteria do not match this script:\n  "
            + "\n  ".join(disagreements)
        )


def load_committed_episodes(path, gate):
    """Load one committed TWAP arm, per episode.

    float_precision="round_trip" is not optional. pandas' default float parser
    is not bit-exact - evaluate_manual_rules_gates.load_twap_episodes documents
    a 4.4e-16 round-trip error on this very file - and Run A's whole question is
    whether 197 episodes come back EXACTLY unchanged. With the default parser
    they would all differ by dust.
    """
    frame = pd.read_csv(path, float_precision="round_trip")
    required = {"gate", "seed", "is_bps", "remaining_quantity"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing column(s) {sorted(missing)}.")

    arm = frame[frame["gate"] == gate]
    if arm.empty:
        raise KeyError(
            f"{path} has no rows for gate '{gate}'. Gates present: "
            f"{sorted(frame['gate'].unique())}."
        )
    return arm[["seed", "is_bps", "remaining_quantity"]].rename(
        columns={
            "is_bps": "committed_is_bps",
            "remaining_quantity": "committed_remaining",
        }
    )


def run_catchup_episode(env, agent, parent_order_size, seed):
    """One reproducible episode of the fill-aware TWAP.

    Structurally the same loop as evaluate_twap_gates.run_twap_episode - same
    seeding, same IS computation - with two differences: the agent is handed the
    outstanding quantity instead of the observation, and the orders it sends are
    counted.
    """
    env.seed(seed)
    env.reset()
    agent.reset()

    total_slippage_reward = 0.0
    orders_sent = 0

    while True:
        # Read before the step, so it reflects the book the agent is deciding
        # against rather than the one left behind afterwards.
        remaining_shares = env.unwrapped.shares_remaining()

        action = agent.predict(remaining_shares)
        if action != agent.ACTION_HOLD:
            orders_sent += 1

        _, _, done, info = env.step(action)
        total_slippage_reward += info["slippage_reward"]
        if done:
            break

    executed_quantity = info["executed_quantity"]
    shortfall_bps = implementation_shortfall_bps(
        total_slippage_reward=total_slippage_reward,
        entry_price=env.unwrapped.entry_price,
        executed_quantity=executed_quantity,
        parent_order_size=parent_order_size,
    )
    return {
        "seed": seed,
        "is_bps": shortfall_bps,
        "remaining_quantity": info["remaining_quantity"],
        "completed": info["remaining_quantity"] == 0,
        "oversold": info["remaining_quantity"] < 0,
        "orders_sent": orders_sent,
        "clamped_orders": env.unwrapped.clamped_order_count,
        "suppressed_orders": env.unwrapped.suppressed_order_count,
    }


def run_arm(arm, episodes, seed_start):
    """Run one arm and return its per-episode frame."""
    config = load_config(PROJECT_ROOT / arm["config"])
    parent_order_size = config["environment"]["parent_order_size"]
    order_fixed_size = config["environment"]["order_fixed_size"]

    env = make_env(config)
    agent = TWAPCatchupAgent(config)

    # How many children a schedule needs when every order fills completely. The
    # committed agent always sends exactly this many; anything above it is a
    # resend.
    baseline_orders = int(np.ceil(parent_order_size / order_fixed_size))

    print(
        f"{arm['name']}: twap_catchup on {arm['gate']}, {episodes} episodes, "
        f"seeds {seed_start}..{seed_start + episodes - 1}"
    )

    rows = []
    for seed in range(seed_start, seed_start + episodes):
        rows.append(run_catchup_episode(env, agent, parent_order_size, seed))
    env.close()

    frame = pd.DataFrame(rows)
    frame["gate"] = arm["gate"]
    frame["extra_orders"] = frame["orders_sent"] - baseline_orders
    return frame, baseline_orders


def check_run_a(frame, committed):
    """Run A: must reproduce the committed arm except on the named seeds."""
    merged = frame.merge(committed, on="seed", how="inner", validate="one_to_one")
    if len(merged) != len(frame):
        raise SystemExit(
            f"Run A pairing is incomplete: {len(frame)} episodes run but "
            f"{len(merged)} matched the committed arm."
        )

    is_difference = (
        merged["is_bps"].to_numpy(dtype=float)
        - merged["committed_is_bps"].to_numpy(dtype=float)
    )
    # Same two-stage defence as paired_test_vs_twap: round-trip parsing above,
    # then snap double-precision dust. IS here is of order 0.2 bps, so 1e-9
    # cannot hide a real difference.
    is_difference[np.abs(is_difference) < IDENTICAL_TOLERANCE_BPS] = 0.0

    merged["is_differs"] = is_difference != 0.0
    merged["remaining_differs"] = (
        merged["remaining_quantity"] != merged["committed_remaining"]
    )
    merged["differs"] = merged["is_differs"] | merged["remaining_differs"]

    differing = merged.loc[merged["differs"], "seed"].tolist()
    unexpected = sorted(set(differing) - set(EXPECTED_DIFFERING_SEEDS))
    missing = sorted(set(EXPECTED_DIFFERING_SEEDS) - set(differing))

    return merged, differing, unexpected, missing


def paired_effect(frame, committed):
    """Run B report-only: paired Wilcoxon of catch-up minus committed TWAP.

    IS is a cost, so a positive difference means the catch-up agent paid MORE.
    That is the direction predicted in advance, so the sign is reported as a
    cost difference rather than flipped into an "improvement".
    """
    merged = frame.merge(committed, on="seed", how="inner", validate="one_to_one")
    differences = (
        merged["is_bps"].to_numpy(dtype=float)
        - merged["committed_is_bps"].to_numpy(dtype=float)
    )
    differences[np.abs(differences) < IDENTICAL_TOLERANCE_BPS] = 0.0

    result = {
        "paired_n": int(len(differences)),
        "n_identical": int(np.sum(differences == 0)),
        "mean_is_difference_bps": float(np.mean(differences)),
        "median_is_difference_bps": float(np.median(differences)),
        "wilcoxon_stat": None,
        "wilcoxon_p": None,
    }
    if np.any(differences != 0):
        statistic, p_value = wilcoxon(differences, alternative="two-sided")
        result["wilcoxon_stat"] = float(statistic)
        result["wilcoxon_p"] = float(p_value)
    return result


def summarise(frame):
    """Report-only summary statistics for one arm."""
    return {
        "gate": frame["gate"].iloc[0],
        "episodes": len(frame),
        "is_mean_bps": float(frame["is_bps"].mean()),
        "is_median_bps": float(frame["is_bps"].median()),
        "is_std_bps": float(frame["is_bps"].std()),
        "completed_pct": float(100.0 * frame["completed"].mean()),
        "mean_unexecuted_shares": float(frame["remaining_quantity"].clip(lower=0).mean()),
        "oversell_episodes": int(frame["oversold"].sum()),
        "mean_orders_sent": float(frame["orders_sent"].mean()),
        "total_extra_orders": int(frame["extra_orders"].clip(lower=0).sum()),
        "episodes_with_extra_orders": int((frame["extra_orders"] > 0).sum()),
        "clamped_orders_total": int(frame["clamped_orders"].sum()),
        "suppressed_orders_total": int(frame["suppressed_orders"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the pre-registered fill-aware TWAP arms on gate_e and "
        "gate_d_varchild."
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Permit seeds from the reserved final-test band. Do not use.",
    )
    parser.add_argument(
        "--criteria",
        type=Path,
        default=PROJECT_ROOT / "configs" / "twap_catchup_criteria.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "twap_catchup_summary.csv",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    verify_criteria_file(args.criteria)

    seed_end = args.seed_start + args.episodes - 1
    assert_seeds_allowed(
        range(args.seed_start, args.seed_start + args.episodes),
        args.final_test,
        context="evaluate_twap_catchup",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Written before the first episode, so the dirty-tree flag describes the
    # tree as the run started rather than as it ended.
    provenance_path = write_config_used(
        args.out.with_name(args.out.stem + "_config_used.yaml"),
        "evaluation/evaluate_twap_catchup.py",
        {
            "seeds": {
                "start": args.seed_start,
                "end": seed_end,
                "episodes_per_arm": args.episodes,
                "band": seed_band_name(args.seed_start, seed_end),
            },
            "final_test": args.final_test,
            "agent": "twap_catchup",
            "criteria_file": str(args.criteria),
            "paired_against": COMMITTED_EPISODES,
            "configs": {
                arm["gate"]: {
                    "path": arm["config"],
                    "environment": load_config(PROJECT_ROOT / arm["config"])[
                        "environment"
                    ],
                }
                for arm in (RUN_A, RUN_B)
            },
        },
    )

    committed_path = PROJECT_ROOT / COMMITTED_EPISODES

    # ------------------------------------------------------------------
    # Run A: regression
    # ------------------------------------------------------------------
    frame_a, baseline_orders_a = run_arm(RUN_A, args.episodes, args.seed_start)
    committed_a = load_committed_episodes(committed_path, RUN_A["committed_gate"])
    merged_a, differing, unexpected, missing = check_run_a(frame_a, committed_a)

    # ------------------------------------------------------------------
    # Run B: effect
    # ------------------------------------------------------------------
    frame_b, baseline_orders_b = run_arm(RUN_B, args.episodes, args.seed_start)
    committed_b = load_committed_episodes(committed_path, RUN_B["committed_gate"])
    effect = paired_effect(frame_b, committed_b)

    summary = pd.DataFrame([summarise(frame_a), summarise(frame_b)])

    episodes_path = args.out.with_name(
        args.out.stem.replace("_summary", "") + "_episodes.csv"
    )
    pd.concat([frame_a, frame_b], ignore_index=True).to_csv(
        episodes_path, index=False
    )
    run_a_path = args.out.with_name(
        args.out.stem.replace("_summary", "") + "_run_a_pairs.csv"
    )
    merged_a.to_csv(run_a_path, index=False)
    summary.to_csv(args.out, index=False)

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Run A - regression on gate_d_varchild")
    print("=" * 72)
    print(f"  episodes differing from the committed arm: {differing}")
    print(f"  pre-registered as expected:                {EXPECTED_DIFFERING_SEEDS}")
    print(f"  unexpected differences:                    {unexpected}")
    print(f"  expected but did NOT differ:               {missing}")
    run_a_pass = len(unexpected) <= 0
    print(f"  VERDICT: {'PASS' if run_a_pass else 'FAIL'}")
    if missing:
        print(
            "  note: a named seed that did not differ is not a failure by the "
            "criteria file, but it means the catch-up had nothing to recover "
            "there and should be explained."
        )
    for seed in EXPECTED_DIFFERING_SEEDS:
        row = merged_a[merged_a["seed"] == seed]
        if row.empty:
            continue
        row = row.iloc[0]
        print(
            f"    seed {seed}: remaining {row['committed_remaining']} -> "
            f"{row['remaining_quantity']}, IS {row['committed_is_bps']:.4f} -> "
            f"{row['is_bps']:.4f} bps"
        )

    print()
    print("=" * 72)
    print("Run B - effect on gate_e")
    print("=" * 72)
    row_b = summary[summary["gate"] == RUN_B["gate"]].iloc[0]
    completion_pass = row_b["completed_pct"] >= MIN_COMPLETED_PCT
    oversell_pass = row_b["oversell_episodes"] <= MAX_OVERSELL_EPISODES
    cost_pass = row_b["is_median_bps"] >= MIN_MEDIAN_IS_BPS

    print(
        f"  completion:   {row_b['completed_pct']:.2f}%  "
        f"(bar {MIN_COMPLETED_PCT}%)  "
        f"{'PASS' if completion_pass else 'FAIL'}"
    )
    print(
        f"  oversell:     {row_b['oversell_episodes']} episode(s)  "
        f"(bar {MAX_OVERSELL_EPISODES})  "
        f"{'PASS' if oversell_pass else 'FAIL'}"
    )
    print(
        f"  median IS:    {row_b['is_median_bps']:.4f} bps  "
        f"(bar {MIN_MEDIAN_IS_BPS})  "
        f"{'PASS' if cost_pass else 'FAIL'}"
    )
    print()
    print("  reported, not gated:")
    print(
        f"    IS mean {row_b['is_mean_bps']:.4f} +/- {row_b['is_std_bps']:.4f} bps, "
        f"mean unexecuted {row_b['mean_unexecuted_shares']:.2f} shares"
    )
    print(
        f"    orders sent per episode: mean {row_b['mean_orders_sent']:.2f} "
        f"against {baseline_orders_b} for a full-fill schedule"
    )
    print(
        f"    resends: {row_b['total_extra_orders']} extra child orders across "
        f"{row_b['episodes_with_extra_orders']} episode(s)"
    )
    print(
        f"    clamped {row_b['clamped_orders_total']}, "
        f"suppressed {row_b['suppressed_orders_total']}"
    )
    print(
        f"    paired vs committed TWAP: median IS difference "
        f"{effect['median_is_difference_bps']:+.4f} bps, mean "
        f"{effect['mean_is_difference_bps']:+.4f} bps "
        f"(positive = the catch-up paid more)"
    )
    if effect["wilcoxon_p"] is None:
        print("    Wilcoxon: every paired difference was exactly zero")
    else:
        print(
            f"    Wilcoxon: stat {effect['wilcoxon_stat']:.1f}, "
            f"p {effect['wilcoxon_p']:.3e}, "
            f"{effect['n_identical']}/{effect['paired_n']} episodes identical"
        )

    failures = frame_b.loc[~frame_b["completed"], "seed"].tolist()
    if failures:
        print(f"    incomplete seeds ({len(failures)}): {failures}")

    print()
    run_b_pass = completion_pass and oversell_pass and cost_pass
    print(
        f"OVERALL: Run A {'PASS' if run_a_pass else 'FAIL'}, "
        f"Run B {'PASS' if run_b_pass else 'FAIL'}"
    )
    print(
        "Next step is decided in advance in "
        f"{Path(args.criteria).name}, section `decision`."
    )

    print()
    print(f"Saved summary to:            {args.out}")
    print(f"Saved per-episode results to:{episodes_path}")
    print(f"Saved Run A pairs to:        {run_a_path}")
    print(f"Saved provenance to:         {provenance_path}")


if __name__ == "__main__":
    main()
