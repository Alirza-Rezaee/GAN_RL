"""Paired Wilcoxon tests for gate_b versus gate_d."""

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "results" / "twap_gate_b&d_episodes.csv"


def main():
    frame = pd.read_csv(INPUT_PATH)

    required_columns = {
        "gate",
        "seed",
        "is_bps",
        "remaining_quantity",
        "completed",
        "oversold",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    gate_b = (
        frame[frame["gate"] == "gate_b"]
        [["seed", "is_bps", "remaining_quantity", "completed", "oversold"]]
        .rename(
            columns={
                "is_bps": "is_b",
                "remaining_quantity": "unexecuted_b",
                "completed": "completed_b",
                "oversold": "oversold_b",
            }
        )
    )

    gate_d = (
        frame[frame["gate"] == "gate_d"]
        [["seed", "is_bps", "remaining_quantity", "completed", "oversold"]]
        .rename(
            columns={
                "is_bps": "is_d",
                "remaining_quantity": "unexecuted_d",
                "completed": "completed_d",
                "oversold": "oversold_d",
            }
        )
    )

    paired = gate_b.merge(
        gate_d,
        on="seed",
        how="inner",
        validate="one_to_one",
    ).sort_values("seed")

    if len(paired) == 0:
        raise ValueError("No shared seeds found.")

    if len(paired) != len(gate_b) or len(paired) != len(gate_d):
        raise ValueError(
            "The two gates do not contain exactly the same seeds."
        )

    print(f"Paired episodes: {len(paired)}")
    print(
        f"Seed range: {paired['seed'].min()}.."
        f"{paired['seed'].max()}"
    )

    # Positive difference means gate_d has lower IS and is therefore better.
    paired["is_difference_b_minus_d"] = paired["is_b"] - paired["is_d"]

    statistic, p_value = wilcoxon(
        paired["is_b"],
        paired["is_d"],
        alternative="two-sided",
        zero_method="wilcox",
    )

    difference = paired["is_difference_b_minus_d"]

    print("\nImplementation Shortfall (IS)")
    print(f"  Gate B mean   : {paired['is_b'].mean():.6f} bps")
    print(f"  Gate D mean   : {paired['is_d'].mean():.6f} bps")
    print(f"  Gate B median : {paired['is_b'].median():.6f} bps")
    print(f"  Gate D median : {paired['is_d'].median():.6f} bps")
    print(f"  Median B - D  : {difference.median():.6f} bps")
    print(f"  D better on   : {(difference > 0).mean() * 100:.2f}% of seeds")
    print(f"  W statistic   : {statistic:.6f}")
    print(f"  p-value       : {p_value:.6f}")

    alpha = 0.05
    if p_value < alpha:
        print(
            "  Conclusion   : significant difference at alpha=0.05."
        )
    else:
        print(
            "  Conclusion   : no statistically significant difference "
            "at alpha=0.05."
        )

    # Unexecuted shares are also a paired numeric metric.
    unexecuted_statistic, unexecuted_p_value = wilcoxon(
        paired["unexecuted_b"],
        paired["unexecuted_d"],
        alternative="two-sided",
        zero_method="wilcox",
    )

    print("\nMean unexecuted shares per paired seed")
    print(
        f"  Gate B mean   : {paired['unexecuted_b'].mean():.6f}"
    )
    print(
        f"  Gate D mean   : {paired['unexecuted_d'].mean():.6f}"
    )
    print(
        f"  W statistic   : {unexecuted_statistic:.6f}"
    )
    print(
        f"  p-value       : {unexecuted_p_value:.6f}"
    )

    output_path = (
        PROJECT_ROOT / "results" / "twap_gate_b&d_wilcoxon_pairs.csv"
    )
    paired.to_csv(output_path, index=False)
    print(f"\nSaved paired data to: {output_path}")


if __name__ == "__main__":
    main()