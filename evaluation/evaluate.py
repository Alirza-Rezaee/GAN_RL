import statistics
from pathlib import Path

import gym
import abides_gym

from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "runs" / "phase1_ppo_tuned_seed44" / "best_model" / "best_model.zip"

NUM_EPISODES = 100
SEED_START = 100
PARENT_ORDER_SIZE = 1000


def create_environment():
    return gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
        order_fixed_size=100,
        parent_order_size=PARENT_ORDER_SIZE,
        execution_window="00:10:00",
        direction="SELL",
    )


def run_episode(model, seed):
    env = create_environment()
    env.seed(seed)
    observation = env.reset()

    total_reward = 0.0
    total_slippage_reward = 0.0
    action_counts = {
    0: 0,
    1: 0,
    2: 0,
    }

    while True:
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)

        action_counts[action] += 1

        observation, reward, done, info = env.step(action)
        total_reward += reward
        total_slippage_reward += info["slippage_reward"]

        

        if done:
            break

    executed_quantity = info["executed_quantity"]
    remaining_quantity = info["remaining_quantity"]

    env.close()

    return {
        "seed": seed,
        "total_reward": total_reward,
        "slippage_reward": total_slippage_reward,
        "executed_quantity": executed_quantity,
        "remaining_quantity": remaining_quantity,
        "fill_rate": executed_quantity / PARENT_ORDER_SIZE,
        "completed": remaining_quantity == 0,
        "action_counts": action_counts,
    }


def main():
    model = PPO.load(str(MODEL_PATH))
    results = []
    total_action_counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    for episode_index in range(NUM_EPISODES):
        seed = SEED_START + episode_index
        result = run_episode(model, seed)
        results.append(result)
        for action, count in result["action_counts"].items():
            total_action_counts[action] += count

        print(
            f"episode={episode_index + 1:03d}, "
            f"seed={seed}, "
            f"total_reward={result['total_reward']:.3f}, "
            f"executed={result['executed_quantity']}, "
            f"remaining={result['remaining_quantity']}"
        )

    rewards = [result["total_reward"] for result in results]
    slippage_rewards = [
        result["slippage_reward"] for result in results
    ]
    executed_quantities = [
        result["executed_quantity"] for result in results
    ]
    fill_rates = [result["fill_rate"] for result in results]
    completed_episodes = [result["completed"] for result in results]

    print("\nPPO summary")
    print(f"episodes: {len(results)}")
    print(
        "total_reward mean/std: "
        f"{statistics.mean(rewards):.3f} / "
        f"{statistics.stdev(rewards):.3f}"
    )
    print(
        "slippage_reward mean/std: "
        f"{statistics.mean(slippage_rewards):.3f} / "
        f"{statistics.stdev(slippage_rewards):.3f}"
    )
    print(
        "executed quantity mean: "
        f"{statistics.mean(executed_quantities):.3f}"
    )
    print(
        "fill rate mean/std: "
        f"{statistics.mean(fill_rates):.3f} / "
        f"{statistics.stdev(fill_rates):.3f}"
    )
    print(
        "completed episodes: "
        f"{sum(completed_episodes)} / {len(results)}"
    )
    total_actions = sum(total_action_counts.values())

    print("action distribution")
    for action in [0, 1, 2]:
        count = total_action_counts[action]
        percentage = 100 * count / total_actions
        print(
            f"action {action}: "
            f"{count} ({percentage:.2f}%)"
        )

if __name__ == "__main__":
    main()