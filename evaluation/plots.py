from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = PROJECT_ROOT / "runs" / "phase1_ppo_tuned_seed44"
EVALUATIONS_PATH = RUN_PATH / "evaluations.npz"

LEARNING_CURVE_PATH = RUN_PATH / "learning_curve.png"
COMPARISON_PATH = RUN_PATH / "twap_vs_ppo.png"


TWAP_REWARD_MEAN = -16.663
TWAP_REWARD_STD = 38.864
TWAP_FILL_RATE_MEAN = 0.989
TWAP_FILL_RATE_STD = 0.034

PPO_REWARD_MEAN = -14.703
PPO_REWARD_STD = 39.506
PPO_FILL_RATE_MEAN = 0.991
PPO_FILL_RATE_STD = 0.031


def plot_learning_curve():
    data = np.load(EVALUATIONS_PATH)

    timesteps = data["timesteps"]
    episode_rewards = data["results"]

    mean_rewards = episode_rewards.mean(axis=1)
    std_rewards = episode_rewards.std(axis=1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        timesteps,
        mean_rewards,
        label="PPO evaluation reward",
        color="tab:blue",
    )

    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="tab:blue",
        alpha=0.2,
        label="Standard deviation",
    )

    plt.axhline(
        y=TWAP_REWARD_MEAN,
        color="tab:orange",
        linestyle="--",
        label="TWAP mean reward",
    )

    plt.xlabel("Training timesteps")
    plt.ylabel("Episode reward")
    plt.title("PPO Learning Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(LEARNING_CURVE_PATH, dpi=150)
    plt.close()

    print(f"Learning curve saved to: {LEARNING_CURVE_PATH}")


def plot_twap_vs_ppo():
    labels = ["TWAP", "PPO tuned"]

    reward_means = [
        TWAP_REWARD_MEAN,
        PPO_REWARD_MEAN,
    ]
    reward_stds = [
        TWAP_REWARD_STD,
        PPO_REWARD_STD,
    ]

    fill_rate_means = [
        TWAP_FILL_RATE_MEAN,
        PPO_FILL_RATE_MEAN,
    ]
    fill_rate_stds = [
        TWAP_FILL_RATE_STD,
        PPO_FILL_RATE_STD,
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(
        labels,
        reward_means,
        yerr=reward_stds,
        capsize=6,
        color=["tab:orange", "tab:blue"],
    )
    axes[0].set_title("Reward Comparison")
    axes[0].set_ylabel("Mean episode reward")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(
        labels,
        fill_rate_means,
        yerr=fill_rate_stds,
        capsize=6,
        color=["tab:orange", "tab:blue"],
    )
    axes[1].set_title("Fill Rate Comparison")
    axes[1].set_ylabel("Mean fill rate")
    axes[1].set_ylim(0.9, 1.0)
    axes[1].grid(axis="y", alpha=0.3)

    figure.suptitle("TWAP vs PPO Tuned")
    figure.tight_layout()

    plt.savefig(COMPARISON_PATH, dpi=150)
    plt.close()

    print(f"Comparison chart saved to: {COMPARISON_PATH}")


if __name__ == "__main__":
    plot_learning_curve()
    plot_twap_vs_ppo()