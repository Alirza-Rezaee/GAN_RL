"""Compare TWAP on several gate configurations using the same market seeds.

Example
-------
    python -m evaluation.evaluate_twap_gates --episodes 200 --seed-start 100

The output contains one summary row per configuration with the metrics needed
to compare the gates.
"""

import argparse
from pathlib import Path

import pandas as pd

from baselines.twap_agent import TWAPAgent
from common.env_factory import PROJECT_ROOT, load_config, make_env
from common.metrics import implementation_shortfall_bps


DEFAULT_CONFIGS = [
    PROJECT_ROOT / "configs" / "gate_b.yaml",
    PROJECT_ROOT / "configs" / "gate_d.yaml",
]


def run_twap_episode(env, agent, parent_order_size, seed):
    """Run one reproducible TWAP episode and return its metrics."""
    env.seed(seed)
    observation = env.reset()
    total_slippage_reward = 0.0

    while True:
        action = agent.predict(observation)
        observation, _, done, info = env.step(action)
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
        "is_bps": shortfall_bps,
        "remaining_quantity": info["remaining_quantity"],
        "completed": info["remaining_quantity"] == 0,
        "oversold": info["remaining_quantity"] < 0,
    }


def evaluate_gate(config_path, episodes, seed_start):
    """Run TWAP for one gate and return its summary metrics."""
    config = load_config(config_path)
    parent_order_size = config["environment"]["parent_order_size"]
    agent = TWAPAgent(config)
    env = make_env(config)
    episode_results = []

    try:
        for episode_index in range(episodes):
            seed = seed_start + episode_index
            agent.reset()
            episode_results.append(
                run_twap_episode(env, agent, parent_order_size, seed)
            )
    finally:
        env.close()

    frame = pd.DataFrame(episode_results)
    frame.insert(0, "seed", range(seed_start, seed_start + len(frame)))
    frame.insert(0, "gate", Path(config_path).stem)
    frame.insert(0, "config", str(Path(config_path)))
    summary = {
        "gate": Path(config_path).stem,
        "config": str(Path(config_path)),
        "episodes": len(frame),
        "seed_start": seed_start,
        "seed_end": seed_start + len(frame) - 1,
        "is_mean_bps": frame["is_bps"].mean(),
        "is_std_bps": frame["is_bps"].std(),
        "completed_pct": frame["completed"].mean() * 100,
        "mean_unexecuted_shares": frame["remaining_quantity"].mean(),
    }
    return summary, frame


def main():
    parser = argparse.ArgumentParser(
        description="Compare TWAP across gate_a, gate_b and gate_c."
    )
    parser.add_argument(
        "--configs",
        nargs=2,
        type=Path,
        default=DEFAULT_CONFIGS,
        metavar=("GATE_B", "GATE_D"),
        help="Two gate YAML files, evaluated in this order.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of episodes per gate (default: 200).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=100,
        help="First seed; every gate uses the same consecutive seeds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "twap_gate_b&d_summary.csv",
        help="Summary CSV path.",
    )
    

    parser.add_argument(
        "--episodes-out",
        type=Path,
        default=PROJECT_ROOT / "results" / "twap_gate_b&d_episodes.csv",
        help="Per-episode results CSV path.",
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    summaries = []
    episode_frames = []
    for config_path in args.configs:
        print(
            f"Evaluating {config_path.stem}: "
            f"{args.episodes} episodes, seeds "
            f"{args.seed_start}..{args.seed_start + args.episodes - 1}",
            flush=True,
        )
        summary, episode_frame = evaluate_gate(config_path, args.episodes, args.seed_start)
        summaries.append(summary)
        episode_frames.append(episode_frame)
        print(
            f"  IS={summary['is_mean_bps']:.3f} +/- "
            f"{summary['is_std_bps']:.3f} bps, "
            f"completed={summary['completed_pct']:.2f}%, "
            f"unexecuted={summary['mean_unexecuted_shares']:.2f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(args.out, index=False)
    all_episodes_frame = pd.concat(episode_frames, ignore_index=True)
    all_episodes_frame.to_csv(args.episodes_out, index=False)
    print(f"\nSaved gate comparison to: {args.out}")
    print(f"Saved per-episode results to: {args.episodes_out}")


if __name__ == "__main__":
    main()
