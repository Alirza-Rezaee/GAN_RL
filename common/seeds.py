"""
Which seed may be used for what.

A seed is a market day. Once a day has been used to develop, tune or decide
anything, a result measured on that same day is partly a memory of it rather than
a prediction. configs/seed_registry.yaml records which days have been spent; this
module is the code that enforces the one band which must stay untouched until the
very end of the thesis.

Why the check cannot live only in env_factory
---------------------------------------------
make_env(config, seed=...) is not the only way a seed reaches the simulator. The
evaluation scripts build the environment once with make_env(config) and then call
env.seed(seed) themselves before every episode, because that is what replays one
specific market day:

    evaluation/evaluate.py:44                      env.seed(seed)
    evaluation/evaluate_twap_gates.py:29           env.seed(seed)
    evaluation/evaluate_manual_rules_gates.py:150  env.seed(seed)

A guard inside make_env() would never see any of those. So each script that
chooses a seed RANGE calls assert_seeds_allowed() once in main(), before any
simulation starts, and make_env() guards only the path it really controls
(execution_agent/train.py:156 and :162, which pass training.seed and
training.eval_seed through it).
"""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "seed_registry.yaml"


class ReservedSeedError(RuntimeError):
    """Raised when a run tries to use the reserved final-test seeds."""


def _band_seeds(band):
    """Every seed a band covers. A band declares either 'range' or 'seeds'."""
    if "range" in band:
        low, high = band["range"]
        return set(range(low, high + 1))
    return set(band["seeds"])


def _validate(registry, registry_path):
    """Two bands claiming the same seed would make the whole registry a lie."""
    owner = {}
    for name, band in registry["bands"].items():
        for seed in _band_seeds(band):
            if seed in owner:
                raise ValueError(
                    f"{registry_path}: seed {seed} is claimed by both "
                    f"'{owner[seed]}' and '{name}'."
                )
            owner[seed] = name

    reserved = registry["enforcement"]["reserved_band"]
    if reserved not in registry["bands"]:
        raise ValueError(
            f"{registry_path}: enforcement.reserved_band is '{reserved}', which "
            f"is not one of {sorted(registry['bands'])}."
        )


def load_seed_registry(path=None):
    registry_path = Path(path) if path else DEFAULT_REGISTRY_PATH
    with registry_path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    _validate(registry, registry_path)
    return registry


def reserved_range(registry=None):
    registry = registry if registry is not None else load_seed_registry()
    band = registry["bands"][registry["enforcement"]["reserved_band"]]
    low, high = band["range"]
    return low, high


def seed_band_name(seed_start, seed_end, registry=None):
    """
    Name of the band covering every seed in [seed_start, seed_end].

    Returns the band name when the whole range sits inside one band, "mixed" when
    it spans more than one band or is only partly registered, and "unregistered"
    when no band claims any of these seeds. The point of the label is to make a
    sloppy range visible in the run record, not to guess what was intended.
    """
    registry = registry if registry is not None else load_seed_registry()
    wanted = set(range(seed_start, seed_end + 1))

    owners = set()
    unclaimed = set(wanted)
    for name, band in registry["bands"].items():
        covered = _band_seeds(band) & wanted
        if covered:
            owners.add(name)
            unclaimed -= covered

    if not owners:
        return "unregistered"
    if len(owners) == 1 and not unclaimed:
        return next(iter(owners))
    return "mixed"


def assert_seeds_allowed(seeds, final_test=False, context="", registry=None):
    """
    Refuse to run on the reserved final-test seeds unless explicitly asked.

    Arguments:
        seeds:      iterable of every seed this run will use.
        final_test: True only when the caller passed --final-test on purpose.
        context:    short description of the caller, used in the message.

    Raises:
        ReservedSeedError if any seed falls in the reserved band and final_test
        is False.
    """
    registry = registry if registry is not None else load_seed_registry()
    low, high = reserved_range(registry)
    hits = sorted(seed for seed in seeds if low <= seed <= high)

    if not hits:
        return

    if final_test:
        # Deliberately loud. Using the test set is a one-way door: the result has
        # to be reported as it comes out, and the run must not be repeated with
        # changed settings afterwards.
        print(
            f"FINAL TEST: {context or 'this run'} is using {len(hits)} reserved "
            f"seed(s) from {low}-{high}. This is the thesis test set. The result "
            f"must be reported as-is and must not be re-run with changes.",
            flush=True,
        )
        return

    flag = registry["enforcement"]["override_flag"]
    shown = hits[:5]
    raise ReservedSeedError(
        f"{context or 'This run'} requested seed(s) {shown}"
        f"{' ...' if len(hits) > len(shown) else ''} from the reserved "
        f"final-test band {low}-{high}. These days are held out for the final "
        f"evaluation of the thesis. Pass {flag} only if that is genuinely what "
        f"you intend."
    )
