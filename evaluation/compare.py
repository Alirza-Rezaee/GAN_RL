"""
Compare a trained agent against the TWAP baseline, with a statistical test.

Why a plain mean comparison is not enough
-----------------------------------------
Episode outcomes in this simulator vary enormously between market days (the
standard deviation of the reward is roughly twenty times the difference we are
trying to detect). Comparing two means without a test would let pure luck look
like a result.

The fix is that both agents are evaluated on the SAME seeds, i.e. the same
simulated market days. That allows a PAIRED test: for every day we compute
(TWAP cost - agent cost) and ask whether that difference is systematically
different from zero. Pairing removes the day-to-day market noise, which is the
dominant source of variance, and is far more sensitive than comparing the two
groups independently.

Example
-------
    python -m evaluation.compare --agent-csv results/ppo.csv --twap-csv results/twap.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

from common.env_factory import PROJECT_ROOT
from common.metrics import price_advantage_bps


def load_paired(agent_csv, twap_csv):
    """
    Load both result files and align them on the seed column.

    Merging on `seed` guarantees we only compare episodes that really are the
    same market day, and it fails loudly if the two evaluations were run with
    different seed ranges.
    """
    agent_frame = pd.read_csv(agent_csv)
    twap_frame = pd.read_csv(twap_csv)

    merged = agent_frame.merge(
        twap_frame,
        on="seed",
        suffixes=("_agent", "_twap"),
        validate="one_to_one",
    )

    if len(merged) == 0:
        raise ValueError(
            "No shared seeds between the two files. Both agents must be "
            "evaluated on the same seed range for a paired comparison."
        )

    if len(merged) != len(agent_frame) or len(merged) != len(twap_frame):
        print(
            f"Warning: only {len(merged)} seeds are shared "
            f"(agent={len(agent_frame)}, twap={len(twap_frame)}). "
            "Comparing the shared subset only."
        )

    # Price Advantage per market day: positive means the agent was cheaper.
    merged["price_advantage_bps"] = [
        price_advantage_bps(agent_is, twap_is)
        for agent_is, twap_is in zip(merged["is_bps_agent"], merged["is_bps_twap"])
    ]

    return merged


def describe(name, values):
    """Print mean, standard deviation and standard error of one metric."""
    mean = values.mean()
    std = values.std()
    # Standard error = how precise the mean estimate is. This is the number that
    # tells you whether your sample size is large enough to see the effect.
    standard_error = std / (len(values) ** 0.5)
    print(f"  {name:<22} {mean:9.3f}  (std {std:7.3f}, se {standard_error:6.3f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent-csv",
        type=str,
        required=True,
        help="CSV produced by evaluation.evaluate for the learned agent.",
    )
    parser.add_argument(
        "--twap-csv",
        type=str,
        default=None,
        help="CSV for TWAP. Defaults to results/twap.csv.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Where to save the merged per-episode comparison CSV.",
    )
    args = parser.parse_args()

    twap_csv = (
        Path(args.twap_csv)
        if args.twap_csv
        else PROJECT_ROOT / "results" / "twap.csv"
    )

    merged = load_paired(Path(args.agent_csv), twap_csv)

    print(f"Paired comparison over {len(merged)} identical market days\n")

    print("Agent:")
    describe("IS (bps)", merged["is_bps_agent"])
    describe("reward", merged["total_reward_agent"])
    describe("fill rate", merged["fill_rate_agent"])

    print("\nTWAP:")
    describe("IS (bps)", merged["is_bps_twap"])
    describe("reward", merged["total_reward_twap"])
    describe("fill rate", merged["fill_rate_twap"])

    print("\nPer-day difference (positive = agent is better):")
    describe("Price Advantage (bps)", merged["price_advantage_bps"])

    # Paired t-test on the per-day differences.
    # H0 (null hypothesis): the mean difference is zero, i.e. the agent is no
    # better than TWAP. A small p-value is evidence against H0.
    t_statistic, p_value = stats.ttest_rel(
        merged["is_bps_twap"],
        merged["is_bps_agent"],
    )

    print("\nPaired t-test on Implementation Shortfall")
    print(f"  t statistic : {t_statistic:.4f}")
    print(f"  p value     : {p_value:.4f}")

    # 0.05 is the conventional threshold in this literature.
    if p_value < 0.05:
        direction = "better" if merged["price_advantage_bps"].mean() > 0 else "worse"
        print(f"  -> Significant at the 5% level: the agent is {direction} than TWAP.")
    else:
        print(
            "  -> NOT significant at the 5% level: on this evidence the agent "
            "cannot be claimed to differ from TWAP."
        )
        print(
            "     Either the effect is genuinely absent, or more evaluation "
            "episodes are needed to detect it."
        )

    output_path = (
        Path(args.out)
        if args.out
        else PROJECT_ROOT / "results" / "comparison.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\nSaved per-episode comparison to: {output_path}")


if __name__ == "__main__":
    main()
