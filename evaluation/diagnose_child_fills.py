"""
Per-child fill diagnostic: where do gate_e's missing shares actually go?

Why this script exists
----------------------
TWAP failed the pre-registered completion criterion on gate_e (78.0%, bar 95%)
and on Candidate B (77.0%). configs/gate_e_validation_criteria.yaml named one
explanation in advance - "the book cannot absorb 5000 shares in 10 minutes" -
and prescribed one response, falling back to Candidate B. That fallback has now
also failed, so the pre-registered decision tree is exhausted and the next move
has to come from a new measurement rather than from the criteria file.

Three facts in the committed results argue the stated explanation is wrong:

  * Candidate B cut the parent order by 40% (5000 -> 3000) and completion did
    not move (78.0% -> 77.0%). If total absorbable volume were the binding
    constraint, a 40% smaller parent order should have helped a great deal.
  * Candidate B also spaces its children FURTHER apart - TWAP sends at steps
    1, 4, 7 there versus 1, 3, 5, 7, 9 on gate_e - so it gives the book more
    time to refill between orders, and is still slightly worse. That is the
    opposite of what book depletion between children predicts.
  * results/impact_probe_summary.csv already measured that a SINGLE 1000-share
    order has a mean fill rate of 0.954 (median 1.0) on an undisturbed book.
    0.954^5 = 0.789, which is within a percentage point of gate_e's observed
    78.0%. Under that model the shortfall is not a capacity wall at all; it is
    ordinary per-order partial fills, five independent chances to lose shares.

The model does NOT fit Candidate B (0.954^3 = 0.868 versus 77.0% observed),
which is exactly why this is a measurement and not a patch. Something beyond
independent per-order partial fills is happening, and guessing at it from two
summary percentages would be guessing.

What this measures
------------------
For every decision step of a TWAP episode it records what the policy asked for,
what the clamp actually sent, and what came back filled:

    requested = min(order_fixed_size, shares_remaining())   # what was sent
    filled    = shares_remaining() before - shares_remaining() after
    shortfall = requested - filled

shares_remaining() is read from common/execution_env.py, which derives it from
the agent's holdings - the same quantity the environment computes done and
reward from, and fresh at step 1 of every episode (see that module's docstring
for why the metrics tracker is not usable here). Bid depth at levels 1-5 is read
at the moment of the decision, before the order goes out, so a shortfall can be
checked against the liquidity that was actually visible.

That turns the open question into a table. If the shortfall is spread roughly
evenly across child indices, it is per-order partial fills and TWAP's refusal to
resend them is the whole story. If it concentrates in the later children, the
book really is being drained and the gate is mis-sized.

What this script does NOT do
----------------------------
It changes nothing and decides nothing. It does not touch baselines/twap_agent.py,
and it writes to its own result files, so every committed gate_e number stays
reproducible.

Usage
-----
    python -m evaluation.diagnose_child_fills --config configs/gate_e.yaml \
        --episodes 30 --seed-start 1000
"""

import argparse
from pathlib import Path

import pandas as pd

from baselines.twap_agent import TWAPAgent
from common.env_factory import PROJECT_ROOT, load_config, make_env
from common.provenance import write_config_used
from common.seeds import assert_seeds_allowed, seed_band_name


# Levels of the bid side to total up, matching evaluation/impact_probe.py so the
# depth numbers here are comparable with the committed probe results.
DEPTH_LEVELS = 5

ACTION_NAMES = {0: "MKT", 1: "LMT", 2: "HOLD"}


def read_bid_depth_total(env, levels=DEPTH_LEVELS):
    """Total visible bid size over the top `levels` price levels, or 0 if the
    book is thinner than that. Same accessor as impact_probe.read_bid_depth."""
    bids = env.unwrapped.gym_agent.parsed_mkt_data["bids"]
    return sum(bids[i][1] if i < len(bids) else 0 for i in range(levels))


def run_instrumented_episode(env, agent, order_fixed_size, seed):
    """
    Run one TWAP episode, returning one record per decision step.

    The episode itself is byte-for-byte the one evaluate_twap_gates.py would
    run: same agent, same seeding, same action sequence. Only the bookkeeping
    around each step is added.
    """
    env.seed(seed)
    env.reset()
    agent.reset()

    records = []
    step_index = 0

    while True:
        step_index += 1

        remaining_before = env.unwrapped.shares_remaining()
        depth_before = read_bid_depth_total(env)
        clamped_before = env.unwrapped.clamped_order_count

        action = agent.predict(None)

        # What the clamp will actually put on the wire. Recomputed here rather
        # than read back from the environment because the environment keeps no
        # per-order record; this mirrors the one line in
        # VariableLastChildExecutionEnv._map_action_space_to_ABIDES_SIMULATOR_SPACE.
        requested = (
            min(order_fixed_size, remaining_before) if action != 2 else 0
        )

        _, _, done, info = env.step(action)

        remaining_after = env.unwrapped.shares_remaining()
        filled = remaining_before - remaining_after

        records.append(
            {
                "seed": seed,
                "step": step_index,
                "action": ACTION_NAMES[action],
                "remaining_before": remaining_before,
                "requested": requested,
                "filled": filled,
                "shortfall": requested - filled,
                "remaining_after": remaining_after,
                "bid_depth_l1_5_before": depth_before,
                "clamped": env.unwrapped.clamped_order_count - clamped_before,
            }
        )

        if done:
            break

    # info["remaining_quantity"] is the environment's own accounting, computed in
    # raw_state_to_done(). Carried out so the diagnostic can be checked against
    # it instead of being trusted: the two must agree.
    return records, info["remaining_quantity"]


def summarise_by_child(child_orders):
    """Shortfall per child order index - the table the whole script exists for.

    `child_index` counts only steps that actually sent an order, so index 1 is
    the first child regardless of which decision step it landed on. That makes
    gate_e (children at steps 1, 3, 5, 7, 9) directly comparable with Candidate B
    (steps 1, 4, 7).
    """
    rows = []
    for child_index, group in child_orders.groupby("child_index"):
        rows.append(
            {
                "child_index": int(child_index),
                "n_orders": len(group),
                "requested_total": int(group["requested"].sum()),
                "filled_total": int(group["filled"].sum()),
                "shortfall_total": int(group["shortfall"].sum()),
                "fill_rate_mean": float(
                    (group["filled"] / group["requested"]).mean()
                ),
                "fill_rate_median": float(
                    (group["filled"] / group["requested"]).median()
                ),
                "partial_fill_pct": float(
                    100.0 * (group["shortfall"] > 0).mean()
                ),
                "bid_depth_l1_5_median": float(
                    group["bid_depth_l1_5_before"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Measure per-child-order fill shortfall for TWAP on one gate."
    )
    parser.add_argument(
        "--config",
        default=PROJECT_ROOT / "configs" / "gate_e.yaml",
        help="gate config to diagnose.",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Permit seeds from the reserved final-test band. Do not use.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "child_fill_diagnostic_summary.csv",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    seed_end = args.seed_start + args.episodes - 1
    assert_seeds_allowed(
        range(args.seed_start, args.seed_start + args.episodes),
        args.final_test,
        context="diagnose_child_fills",
    )

    config = load_config(args.config)
    environment_config = config["environment"]
    order_fixed_size = environment_config["order_fixed_size"]
    parent_order_size = environment_config["parent_order_size"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = write_config_used(
        args.out.with_name(args.out.stem + "_config_used.yaml"),
        "evaluation/diagnose_child_fills.py",
        {
            "seeds": {
                "start": args.seed_start,
                "end": seed_end,
                "episodes": args.episodes,
                "band": seed_band_name(args.seed_start, seed_end),
            },
            "final_test": args.final_test,
            "agent": "twap",
            "question": (
                "Does gate_e's completion shortfall come from per-order partial "
                "fills, or from the book being drained by the later children?"
            ),
            "configs": {
                Path(args.config).stem: {
                    "path": str(args.config),
                    "environment": environment_config,
                }
            },
        },
    )

    env = make_env(config)
    agent = TWAPAgent(config)

    print(
        f"Diagnosing {Path(args.config).stem}: {args.episodes} episodes, "
        f"seeds {args.seed_start}..{seed_end}, "
        f"order_fixed_size={order_fixed_size}, parent={parent_order_size}"
    )

    all_steps = []
    episode_rows = []

    for seed in range(args.seed_start, args.seed_start + args.episodes):
        records, env_remaining = run_instrumented_episode(
            env, agent, order_fixed_size, seed
        )
        all_steps.extend(records)

        frame = pd.DataFrame(records)
        orders = frame[frame["requested"] > 0]
        episode_rows.append(
            {
                "seed": seed,
                "n_orders_sent": len(orders),
                "requested_total": int(orders["requested"].sum()),
                "filled_total": int(orders["filled"].sum()),
                "shortfall_total": int(orders["shortfall"].sum()),
                "n_partial_fills": int((orders["shortfall"] > 0).sum()),
                "remaining_diagnostic": int(frame["remaining_after"].iloc[-1]),
                "remaining_env": int(env_remaining),
                "completed": int(env_remaining) == 0,
            }
        )
        print(
            f"  seed {seed}: sent {len(orders)} children, "
            f"shortfall {episode_rows[-1]['shortfall_total']:>5}, "
            f"{episode_rows[-1]['n_partial_fills']} partial fill(s), "
            f"{'completed' if episode_rows[-1]['completed'] else 'INCOMPLETE'}"
        )

    env.close()

    steps = pd.DataFrame(all_steps)
    episodes = pd.DataFrame(episode_rows)

    # child_index: 1 for the first order actually sent in the episode, 2 for the
    # second, and so on. Only sending steps get one.
    child_orders = steps[steps["requested"] > 0].copy()
    child_orders["child_index"] = (
        child_orders.groupby("seed").cumcount() + 1
    )

    summary = summarise_by_child(child_orders)

    steps_path = args.out.with_name(args.out.stem.replace("_summary", "") + "_steps.csv")
    episodes_path = args.out.with_name(
        args.out.stem.replace("_summary", "") + "_episodes.csv"
    )
    steps.to_csv(steps_path, index=False)
    episodes.to_csv(episodes_path, index=False)
    summary.to_csv(args.out, index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    disagreements = episodes[
        episodes["remaining_diagnostic"] != episodes["remaining_env"]
    ]
    print()
    if len(disagreements) == 0:
        print(
            "Accounting check: shares_remaining() agrees with the environment's "
            f"own remaining_quantity on all {len(episodes)} episodes."
        )
    else:
        print(
            f"WARNING: the diagnostic and the environment disagree on "
            f"{len(disagreements)} episode(s). Every number below is suspect."
        )
        print(disagreements.to_string(index=False))

    print()
    print("Shortfall by child order index:")
    print(
        f"{'child':>6} {'orders':>7} {'requested':>10} {'filled':>8} "
        f"{'short':>7} {'fill% med':>10} {'fill% mean':>11} "
        f"{'partial%':>9} {'depth med':>10}"
    )
    for row in summary.itertuples():
        print(
            f"{row.child_index:>6} {row.n_orders:>7} {row.requested_total:>10} "
            f"{row.filled_total:>8} {row.shortfall_total:>7} "
            f"{100 * row.fill_rate_median:>10.1f} "
            f"{100 * row.fill_rate_mean:>11.1f} "
            f"{row.partial_fill_pct:>9.1f} {row.bid_depth_l1_5_median:>10.0f}"
        )

    completed_pct = 100.0 * episodes["completed"].mean()
    print()
    print(
        f"Episodes completed: {completed_pct:.1f}%  "
        f"({int(episodes['completed'].sum())}/{len(episodes)})"
    )
    print(
        f"Mean shortfall per episode: "
        f"{episodes['shortfall_total'].mean():.1f} shares of {parent_order_size}"
    )
    print(
        f"Episodes with at least one partial fill: "
        f"{100.0 * (episodes['n_partial_fills'] > 0).mean():.1f}%"
    )

    print()
    print(f"Saved per-child summary to:  {args.out}")
    print(f"Saved per-step records to:   {steps_path}")
    print(f"Saved per-episode totals to: {episodes_path}")
    print(f"Saved provenance to:         {provenance_path}")


if __name__ == "__main__":
    main()
