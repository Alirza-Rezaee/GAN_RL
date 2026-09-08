"""
Train the phase-1 PPO baseline agent on markets-execution-v0.

Three things this script deliberately does, which the first version did not:

1. It saves a copy of the exact config into the run directory, so a result can
   always be traced back to the hyperparameters that produced it.
2. It refuses to write into a run directory that already exists, so one run can
   never silently overwrite another.
3. It re-seeds the evaluation environment before every evaluation, so the
   learning curve reflects the policy improving rather than the market changing.

Examples
--------
    python -m execution_agent.train --run-name smoke_test
    python -m execution_agent.train --full --seed 42 --run-name phase1_ppo_seed42
"""

import argparse
import shutil
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from common.env_factory import (
    DEFAULT_CONFIG_PATH,
    PROJECT_ROOT,
    load_config,
    make_env,
)


RUNS_PATH = PROJECT_ROOT / "runs"


class ReseedingEvalCallback(EvalCallback):
    """
    EvalCallback that replays the same fixed set of market days every time.

    Problem with the default behaviour: the evaluation environment keeps drawing
    new random market days, so consecutive evaluation points differ both because
    the policy changed and because the market changed. With this simulator the
    market effect is much larger than the policy effect, which makes the learning
    curve almost unreadable and makes "best model" selection close to random.

    Fix: re-seed the evaluation environment with the same seed just before each
    evaluation, so every evaluation point is measured on an identical set of days
    and the only thing that varies is the policy.
    """

    def __init__(self, *args, eval_seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_seed = eval_seed

    def _on_step(self):
        # Same trigger condition as the parent class: act only on evaluation
        # steps, and re-seed just before the parent runs the episodes.
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_env.seed(self.eval_seed)

        return super()._on_step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full training budget instead of the short smoke test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training seed from the config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help=(
            "Directory name under runs/ for this training run. Required, and "
            "must be unique, so that runs cannot overwrite each other."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file to use. Defaults to configs/phase1_baseline.yaml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing run directory of the same name and start over.",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config = load_config(config_path)

    algorithm_config = config["algorithm"]
    training_config = config["training"]

    # Only PPO is implemented. Without this check, setting name: "dqn" in the
    # YAML would silently train PPO anyway and the config would be a lie.
    algorithm_name = algorithm_config["name"].lower()
    if algorithm_name != "ppo":
        raise ValueError(
            f"Config requests algorithm '{algorithm_name}', but only 'ppo' is "
            "implemented in this script."
        )

    seed = args.seed if args.seed is not None else training_config["seed"]
    total_timesteps = (
        training_config["total_timesteps"]
        if args.full
        else training_config["quick_test_timesteps"]
    )

    # ---- Run directory, protected against accidental overwrites -------------
    run_path = RUNS_PATH / args.run_name

    if run_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_path}\n"
                "Choose a different --run-name, or pass --overwrite to discard "
                "the previous run."
            )
        shutil.rmtree(run_path)

    best_model_path = run_path / "best_model"
    log_path = run_path / "tensorboard"
    best_model_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    # ---- Reproducibility snapshot ------------------------------------------
    # Save the config as it was actually used, with the command-line overrides
    # already applied, so this file alone is enough to reproduce the run.
    snapshot = dict(config)
    snapshot["training"] = dict(training_config)
    snapshot["training"]["seed"] = seed
    snapshot["training"]["timesteps_used"] = total_timesteps
    snapshot["training"]["source_config_file"] = str(config_path)

    with (run_path / "config_used.yaml").open("w", encoding="utf-8") as snapshot_file:
        yaml.safe_dump(snapshot, snapshot_file, allow_unicode=True, sort_keys=False)

    # ---- Environments ------------------------------------------------------
    # Monitor with a filename writes one row per training episode to
    # monitor.csv, so the raw training rewards survive on disk.
    train_env = Monitor(make_env(config, seed=seed), filename=str(run_path / "monitor"))

    # The evaluation environment uses its own seed range, kept separate from
    # both the training seed and the final test seeds. Selecting the best model
    # on the final test seeds would leak the test set into model selection.
    eval_seed = training_config["eval_seed"]
    eval_env = Monitor(make_env(config, seed=eval_seed))

    eval_callback = ReseedingEvalCallback(
        eval_env,
        eval_seed=eval_seed,
        best_model_save_path=str(best_model_path),
        log_path=str(run_path),
        eval_freq=training_config["eval_freq"],
        # More episodes per evaluation means a less noisy estimate, which matters
        # a lot here: with 5 episodes the noise is larger than the effect being
        # measured, so "best model" would mostly be picking lucky days.
        n_eval_episodes=training_config["n_eval_episodes"],
        deterministic=True,
        render=False,
    )

    # ---- Model -------------------------------------------------------------
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

    print(f"Training run '{args.run_name}': seed={seed}, timesteps={total_timesteps}")

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(str(run_path / "final_model"))

    train_env.close()
    eval_env.close()

    print(f"\nTraining finished: {total_timesteps} timesteps")
    print(f"Run directory: {run_path}")
    print(f"  best model : {best_model_path / 'best_model.zip'}")
    print(f"  config used: {run_path / 'config_used.yaml'}")


if __name__ == "__main__":
    main()
