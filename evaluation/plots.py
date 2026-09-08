"""
Draw the two figures for phase 1.

Every number is read from the CSV files produced by evaluation.evaluate and from
the run's evaluations.npz. Nothing is typed in by hand, so re-running an
evaluation automatically updates the figures instead of leaving stale values
behind.

Example
-------
    python -m evaluation.plots --run runs/<run_name>
"""

import argparse
from pathlib import Path

import matplotlib

# "Agg" renders to files without needing a GUI window, which is what we want for
# a script that only saves PNGs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.env_factory import PROJECT_ROOT


def plot_learning_curve(run_path, twap_reward_mean, output_path):
    """
    Plot the periodic evaluation reward recorded during training.

    evaluations.npz is written by SB3's EvalCallback. Its "results" array has one
    row per evaluation point and one column per evaluation episode.
    """
    evaluations_path = run_path / "evaluations.npz"
    if not evaluations_path.exists():
        print(f"Skipping learning curve: {evaluations_path} not found")
        return

    data = np.load(evaluations_path)
    timesteps = data["timesteps"]
    episode_rewards = data["results"]

    mean_rewards = episode_rewards.mean(axis=1)
    std_rewards = episode_rewards.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label="PPO evaluation reward", color="tab:blue")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="tab:blue",
        alpha=0.2,
        label="Standard deviation across eval episodes",
    )

    if twap_reward_mean is not None:
        plt.axhline(
            y=twap_reward_mean,
            color="tab:orange",
            linestyle="--",
            label=f"TWAP mean reward ({twap_reward_mean:.2f})",
        )

    plt.xlabel("Training timesteps")
    plt.ylabel("Episode reward")
    plt.title(f"PPO Learning Curve ({run_path.name})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Learning curve saved to: {output_path}")


def plot_comparison(agent_frame, twap_frame, agent_label, output_path):
    """
    Bar chart of Implementation Shortfall and fill rate, with error bars.

    The error bars are standard deviations across evaluation episodes, so they
    show how noisy the simulator is, not how uncertain the mean is.
    """
    labels = ["TWAP", agent_label]

    is_means = [twap_frame["is_bps"].mean(), agent_frame["is_bps"].mean()]
    is_stds = [twap_frame["is_bps"].std(), agent_frame["is_bps"].std()]

    fill_means = [twap_frame["fill_rate"].mean(), agent_frame["fill_rate"].mean()]
    fill_stds = [twap_frame["fill_rate"].std(), agent_frame["fill_rate"].std()]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["tab:orange", "tab:blue"]

    axes[0].bar(labels, is_means, yerr=is_stds, capsize=6, color=colors)
    axes[0].set_title("Implementation Shortfall (lower is better)")
    axes[0].set_ylabel("Mean IS (bps)")
    axes[0].axhline(y=0, color="black", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(labels, fill_means, yerr=fill_stds, capsize=6, color=colors)
    axes[1].set_title("Fill Rate (higher is better)")
    axes[1].set_ylabel("Mean fill rate")
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle(f"TWAP vs {agent_label}")
    figure.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Comparison chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Training run directory, e.g. runs/phase1_ppo_seed42.",
    )
    parser.add_argument(
        "--agent-csv",
        type=str,
        default=None,
        help="Agent results CSV. Defaults to results/ppo.csv.",
    )
    parser.add_argument(
        "--twap-csv",
        type=str,
        default=None,
        help="TWAP results CSV. Defaults to results/twap.csv.",
    )
    parser.add_argument(
        "--agent-label",
        type=str,
        default="PPO",
        help="Label used for the agent in the figures.",
    )
    args = parser.parse_args()

    run_path = Path(args.run)
    results_path = PROJECT_ROOT / "results"

    agent_csv = Path(args.agent_csv) if args.agent_csv else results_path / "ppo.csv"
    twap_csv = Path(args.twap_csv) if args.twap_csv else results_path / "twap.csv"

    agent_frame = pd.read_csv(agent_csv)
    twap_frame = pd.read_csv(twap_csv)

    plot_learning_curve(
        run_path=run_path,
        twap_reward_mean=twap_frame["total_reward"].mean(),
        output_path=run_path / "learning_curve.png",
    )
    plot_comparison(
        agent_frame=agent_frame,
        twap_frame=twap_frame,
        agent_label=args.agent_label,
        output_path=run_path / "twap_vs_agent.png",
    )


if __name__ == "__main__":
    main()
