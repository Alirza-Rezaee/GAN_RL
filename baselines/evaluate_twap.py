import statistics

import gym
import abides_gym

from baselines.twap_agent import TWAPAgent


NUM_EPISODES = 100
SEED_START = 100


def run_episode(seed):
    env = gym.make(
        "markets-execution-v0",
        background_config="rmsc04",
        order_fixed_size=100,
        parent_order_size=1000,
        execution_window="00:10:00",
        direction="SELL",
    )

    env.seed(seed)
    observation = env.reset()

    agent = TWAPAgent()
    total_reward = 0.0
    total_slippage_reward = 0.0

    while True:
        action = agent.predict(observation)
        observation, reward, done, info = env.step(action)

        total_reward += reward
        total_slippage_reward += info["slippage_reward"]

        if done:
            break

    executed_quantity = info["executed_quantity"]
    remaining_quantity = info["remaining_quantity"]
    fill_rate = executed_quantity / 1000

    env.close()

    return {
        "seed": seed,
        "total_reward": total_reward,
        "slippage_reward": total_slippage_reward,
        "executed_quantity": executed_quantity,
        "remaining_quantity": remaining_quantity,
        "fill_rate": fill_rate,
        "completed": remaining_quantity == 0,
    }

def main():
    results = []

    for episode_index in range(NUM_EPISODES):
        seed = SEED_START + episode_index
        result = run_episode(seed)
        results.append(result)

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
    fill_rates = [
        result["fill_rate"] for result in results
    ]

    completed_episodes = [
        result["completed"] for result in results
    ]
    print("\nTWAP summary")
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
if __name__ == "__main__":
    main()