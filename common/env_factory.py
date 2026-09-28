"""
Single place where the YAML config is read and the ABIDES gym environment is
built.

Why this file exists
--------------------
Previously the environment parameters were written by hand in train.py, in
evaluate.py and in the TWAP scripts. Each copy passed a different subset of
parameters, so the remaining ones silently fell back to the library defaults.
Training and evaluation happened to match only because the defaults happened to
equal the config values. The moment one number changed in the YAML, the agent
would have been trained on one market and evaluated on another one, with no
error message.

Every script in this project must now build its environment through
make_env() so that there is exactly one source of truth: the config file.
"""

from pathlib import Path

import gym
import yaml

# abides_gym must be imported for its side effect: importing it is what
# registers the "markets-execution-v0" id inside gym.
import abides_gym  # noqa: F401

from common.seeds import assert_seeds_allowed

# Imported for its side effect as well: importing it registers the variant gym id.
from common.execution_env import VARIANT_ENV_ID


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase1_baseline.yaml"

# The exact set of keys under `environment:` that are forwarded to the gym
# environment constructor. Listed explicitly so that a typo in the YAML raises
# an error instead of being silently ignored.
ENV_PARAM_KEYS = [
    "background_config",
    "mkt_close",
    "timestep_duration",
    "starting_cash",
    "order_fixed_size",
    "parent_order_size",
    "execution_window",
    "direction",
    "state_history_length",
    "market_data_buffer_length",
    "first_interval",
]

# Optional `env_variant` key under `environment:`. A config that does not set it
# resolves to "stock", which uses environment_config["id"] unchanged - so every
# config written before this key existed keeps its behaviour byte-for-byte.
# `env_variant` is deliberately absent from ENV_PARAM_KEYS, so it is never
# forwarded to the environment constructor.
ENV_VARIANTS = {
    "stock": None,
    "variable_last_child": VARIANT_ENV_ID,
}


def load_config(config_path=None):
    """Read the YAML config file and return it as a plain dict."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def make_env(config, seed=None, allow_final_test=False):
    """
    Build one ABIDES execution environment from a loaded config dict.

    Arguments:
        config: the full config dict returned by load_config().
        seed:   if given, seeds the environment's random generator. The seed
                controls which simulated market day reset() produces, so the
                same seed always replays the same day.
        allow_final_test: permit a seed from the reserved final-test band. Must
                never be set by training code - selecting a model on test days
                would leak the test set into model selection.

    Returns:
        a gym environment, ready for reset().
    """
    environment_config = config["environment"]

    # Fail loudly if the YAML is missing a key, instead of letting the library
    # default silently take over.
    missing_keys = [
        key for key in ENV_PARAM_KEYS if key not in environment_config
    ]
    if missing_keys:
        raise KeyError(
            f"Config section 'environment' is missing keys: {missing_keys}"
        )

    env_kwargs = {key: environment_config[key] for key in ENV_PARAM_KEYS}

    variant = environment_config.get("env_variant", "stock")
    if variant not in ENV_VARIANTS:
        raise ValueError(
            f"Unknown env_variant '{variant}'. Choices: {sorted(ENV_VARIANTS)}"
        )

    variant_env_id = ENV_VARIANTS[variant]
    if variant_env_id is None:
        env_id = environment_config["id"]
    else:
        if environment_config["id"] != "markets-execution-v0":
            raise ValueError(
                f"env_variant '{variant}' is a subclass of markets-execution-v0, but "
                f"the config asks for id '{environment_config['id']}'."
            )
        env_id = variant_env_id

    env = gym.make(env_id, **env_kwargs)

    if seed is not None:
        # Guards only the seeds that reach the simulator through make_env, which
        # in practice means execution_agent/train.py. The evaluation scripts seed
        # per episode and call assert_seeds_allowed() in their own main().
        assert_seeds_allowed(
            [seed],
            allow_final_test,
            context="make_env(seed=...)",
        )
        env.seed(seed)

    return env
