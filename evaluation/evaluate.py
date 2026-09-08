"""
Run one agent for N validation episodes and save the per-episode metrics to CSV.

One script handles both agents (TWAP and a trained PPO model) so that they are
guaranteed to be measured on exactly the same market days with exactly the same
environment parameters. The CSV output is what the comparison and plotting
scripts read, so no number ever has to be copied by hand.

Examples
--------
    python -m evaluation.evaluate --agent twap
    python -m evaluation.evaluate --agent ppo --model runs/<run_name>/best_model/best_model.zip
"""

import argparse
from pathlib import Path

# torch must be imported before pandas on Windows. Both ship their own copy of
# the Intel OpenMP runtime, and if pandas loads first, torch's DLL fails to
# initialise with "WinError 1114". Importing torch first avoids the clash.
import torch  # noqa: F401
import pandas as pd
from stable_baselines3 import PPO

from baselines.twap_agent import TWAPAgent
from common.env_factory import PROJECT_ROOT, load_config, make_env
from common.metrics import implementation_shortfall_bps


def run_episode(env, agent_step, parent_order_size, seed):
    """
    Play one full episode and return its metrics.

    Arguments:
        env:               the environment, reused across episodes.
        agent_step:        function(observation) -> action id.
        parent_order_size: shares the agent must execute.
        seed:              seed of the market day to replay. Seeding before each
                           reset() makes the episode exactly reproducible, which
                           is what lets us compare two agents on the same days.
    """
    # Seed first, then reset: reset() draws the simulation seed from the
    # generator that seed() initialises.
    env.seed(seed)
    observation = env.reset()

    total_reward = 0.0
    total_slippage_reward = 0.0
    action_counts = {0: 0, 1: 0, 2: 0}

    while True:
        action = agent_step(observation)
        action_counts[action] += 1

        observation, reward, done, info = env.step(action)

        # total_reward includes the end-of-episode penalty for unexecuted shares;
        # slippage_reward is the pure trading PnL, which is what feeds into IS.
        total_reward += reward
        total_slippage_reward += info["slippage_reward"]

        if done:
            break

    executed_quantity = info["executed_quantity"]

    # The arrival price is stored on the environment itself, and our env fix
    # guarantees it belongs to this episode's market day.
    entry_price = env.unwrapped.entry_price

    shortfall_bps = implementation_shortfall_bps(
        total_slippage_reward=total_slippage_reward,
        entry_price=entry_price,
        executed_quantity=executed_quantity,
        parent_order_size=parent_order_size,
    )

    return {
        "seed": seed,
        "total_reward": total_reward,
        "slippage_reward": total_slippage_reward,
        "late_penalty_reward": info["late_penalty_reward"],
        "entry_price": entry_price,
        "executed_quantity": executed_quantity,
        "remaining_quantity": info["remaining_quantity"],
        "fill_rate": executed_quantity / parent_order_size,
        "is_bps": shortfall_bps,
        "completed": info["remaining_quantity"] == 0,
        "action_mkt": action_counts[0],
        "action_lmt": action_counts[1],
        "action_hold": action_counts[2],
    }


def build_agent_step(agent_name, config, model_path):
    """
    Return a function(observation) -> action for the requested agent, plus an
    optional per-episode reset hook.
    """
    if agent_name == "twap":
        agent = TWAPAgent(config)

        def agent_step(observation):
            return agent.predict(observation)

        return agent_step, agent.reset

    if agent_name == "ppo":
        if model_path is None:
            raise ValueError("--model is required when --agent ppo is used")

        model = PPO.load(str(model_path))

        def agent_step(observation):
            # deterministic=True disables exploration noise: at evaluation time
            # we want the policy's best action, not a sample from it.
            action, _ = model.predict(observation, deterministic=True)
            return int(action)

        # A neural network policy has no per-episode state to clear.
        return agent_step, lambda: None

    raise ValueError(f"Unknown agent: {agent_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["twap", "ppo"],
        required=True,
        help="Which agent to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the saved PPO model (required for --agent ppo).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file to use. Defaults to configs/phase1_baseline.yaml.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path. Defaults to results/<agent>.csv.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Evaluation seeds come from the config, so training and evaluation can
    # never accidentally drift apart.
    evaluation_config = config["evaluation"]
    num_episodes = evaluation_config["episodes"]
    seed_start = evaluation_config["seed_start"]
    parent_order_size = config["environment"]["parent_order_size"]

    output_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "results" / f"{args.agent}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agent_step, agent_reset = build_agent_step(args.agent, config, args.model)

    # One environment is created and reused; re-seeding before each reset gives
    # a fresh, reproducible market day without rebuilding the whole simulator.
    env = make_env(config)
    results = []

    for episode_index in range(num_episodes):
        seed = seed_start + episode_index
        agent_reset()
        result = run_episode(env, agent_step, parent_order_size, seed)
        results.append(result)

        print(
            f"episode={episode_index + 1:03d}/{num_episodes} "
            f"seed={seed} "
            f"reward={result['total_reward']:8.3f} "
            f"IS={result['is_bps']:7.2f}bps "
            f"executed={result['executed_quantity']}"
        )

    env.close()

    frame = pd.DataFrame(results)
    frame.to_csv(output_path, index=False)

    print(f"\n{args.agent.upper()} summary over {len(frame)} episodes")
    print(f"  IS (bps)    mean/std : {frame['is_bps'].mean():.2f} / {frame['is_bps'].std():.2f}")
    print(f"  reward      mean/std : {frame['total_reward'].mean():.3f} / {frame['total_reward'].std():.3f}")
    print(f"  fill rate   mean/std : {frame['fill_rate'].mean():.4f} / {frame['fill_rate'].std():.4f}")
    print(f"  completed episodes   : {int(frame['completed'].sum())} / {len(frame)}")
    print(f"\nSaved per-episode results to: {output_path}")


if __name__ == "__main__":
    main()
