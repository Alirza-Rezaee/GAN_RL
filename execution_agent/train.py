import argparse
from pathlib import Path

import gym
import yaml
import abides_gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_baseline.yaml"
RUNS_PATH = PROJECT_ROOT / "runs"


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def create_environment(environment_config):
    return gym.make(
        environment_config["id"],
        background_config=environment_config["background_config"],
        mkt_close=environment_config["mkt_close"],
        timestep_duration=environment_config["timestep_duration"],
        starting_cash=environment_config["starting_cash"],
        order_fixed_size=environment_config["order_fixed_size"],
        parent_order_size=environment_config["parent_order_size"],
        execution_window=environment_config["execution_window"],
        direction=environment_config["direction"],
        state_history_length=environment_config["state_history_length"],
        market_data_buffer_length=environment_config[
            "market_data_buffer_length"
        ],
        first_interval=environment_config["first_interval"],
    )


def main(quick_test, seed_override=None, run_name="phase1_ppo"):
    config = load_config()

    environment_config = config["environment"]
    algorithm_config = config["algorithm"]
    training_config = config["training"]

    seed = (
        seed_override
        if seed_override is not None
        else training_config["seed"]
    )
    total_timesteps = (
        training_config["quick_test_timesteps"]
        if quick_test
        else training_config["total_timesteps"]
    )

    run_path = RUNS_PATH / run_name
    best_model_path = run_path / "best_model"
    log_path = run_path / "tensorboard"

    best_model_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    train_env = Monitor(create_environment(environment_config))
    eval_env = Monitor(create_environment(environment_config))

    train_env.seed(seed)
    eval_env.seed(seed + 1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_path),
        log_path=str(run_path),
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model = PPO(
        algorithm_config["policy"],
        train_env,
        learning_rate=algorithm_config["learning_rate"],
        n_steps=algorithm_config["n_steps"],
        batch_size=algorithm_config["batch_size"],
        gamma=algorithm_config["gamma"],
        ent_coef=algorithm_config["ent_coef"],
        verbose=1,
        seed=seed,
        tensorboard_log=str(log_path),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    model.save(str(run_path / "final_model"))

    train_env.close()
    eval_env.close()

    print(f"Training finished: {total_timesteps} timesteps")
    print(f"Models and logs saved in: {run_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full training budget instead of the quick test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training seed.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="phase1_ppo",
        help="Directory name for saving this training run.",
    )

    args = parser.parse_args()

    main(
        quick_test=not args.full,
        seed_override=args.seed,
        run_name=args.run_name,
    )