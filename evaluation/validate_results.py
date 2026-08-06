"""
validate_results.py

Validates evaluation/simulation_results.csv (produced by
evaluation/export_results.py), checks for data quality issues, and
exports a cleaned dataset ready for dashboard visualization.

Checks performed:
1. File loads correctly and has all expected columns
2. No missing (NaN/empty) values in any required field
3. No duplicate (episode_number, policy_name) records
4. Policy output consistency:
   - revenue >= 0 (can't earn negative dollars)
   - 0 <= inventory_remaining <= initial_inventory
   - selling_price > 0
5. Every policy present has a consistent episode count (no partial/
   truncated simulation runs silently mixed into the dataset)

Usage
-----
    python -m evaluation.validate_results
    python -m evaluation.validate_results --csv path/to/results.csv --initial-inventory 100
"""

import os
import argparse

import pandas as pd


REQUIRED_COLUMNS = [
    "episode_number", "policy_name", "revenue", "reward",
    "inventory_remaining", "selling_price",
]


def validate_results(csv_path: str, initial_inventory: int = 100) -> dict:
    """
    Runs all validation checks against the CSV at `csv_path`.

    Returns
    -------
    dict
        {
            "df": the loaded (and, if needed, cleaned) DataFrame,
            "issues": list of human-readable issue descriptions found,
            "is_clean": bool, True if zero issues were found,
        }
    """
    issues = []

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Simulation results file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # --- Check 1: expected columns present ---
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required column(s): {missing_cols}")

    # --- Check 2: missing values ---
    for col in REQUIRED_COLUMNS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            issues.append(f"{n_missing} missing value(s) found in column '{col}'")

    # --- Check 3: duplicate (episode_number, policy_name) records ---
    dup_mask = df.duplicated(subset=["episode_number", "policy_name"], keep=False)
    n_duplicates = dup_mask.sum()
    if n_duplicates > 0:
        issues.append(
            f"{n_duplicates} duplicate record(s) found for the same "
            f"(episode_number, policy_name) pair"
        )

    # --- Check 4: policy output consistency ---
    n_negative_revenue = (df["revenue"] < 0).sum()
    if n_negative_revenue > 0:
        issues.append(f"{n_negative_revenue} record(s) with negative revenue (invalid)")

    n_bad_inventory = ((df["inventory_remaining"] < 0) | (df["inventory_remaining"] > initial_inventory)).sum()
    if n_bad_inventory > 0:
        issues.append(
            f"{n_bad_inventory} record(s) with inventory_remaining outside "
            f"the valid range [0, {initial_inventory}]"
        )

    n_bad_price = (df["selling_price"] <= 0).sum()
    if n_bad_price > 0:
        issues.append(f"{n_bad_price} record(s) with non-positive selling_price (invalid)")

    # --- Check 5: consistent episode count per policy ---
    counts_per_policy = df.groupby("policy_name")["episode_number"].count()
    if counts_per_policy.nunique() > 1:
        issues.append(
            f"Inconsistent episode counts across policies: "
            f"{counts_per_policy.to_dict()} - one or more policies may have a "
            f"truncated/incomplete simulation run"
        )

    # --- Build the cleaned dataset: drop any row involved in an issue above ---
    clean_mask = (
        df[REQUIRED_COLUMNS].notna().all(axis=1)
        & ~dup_mask
        & (df["revenue"] >= 0)
        & (df["inventory_remaining"] >= 0)
        & (df["inventory_remaining"] <= initial_inventory)
        & (df["selling_price"] > 0)
    )
    clean_df = df[clean_mask].reset_index(drop=True)

    n_dropped = len(df) - len(clean_df)
    if n_dropped > 0:
        issues.append(f"{n_dropped} total row(s) dropped from the cleaned export due to the issues above")

    return {
        "df": clean_df,
        "raw_row_count": len(df),
        "clean_row_count": len(clean_df),
        "issues": issues,
        "is_clean": len(issues) == 0,
        "counts_per_policy": counts_per_policy.to_dict(),
    }


def export_validated_csv(clean_df: pd.DataFrame, output_path: str) -> str:
    """Exports the cleaned DataFrame to CSV, creating parent directories if needed."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simulation results and export a cleaned dataset.")
    parser.add_argument("--csv", type=str, default="evaluation/simulation_results.csv")
    parser.add_argument("--output", type=str, default="evaluation/validated_results.csv")
    parser.add_argument("--initial-inventory", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Validating '{args.csv}'...")
    result = validate_results(args.csv, initial_inventory=args.initial_inventory)

    print(f"\nRaw rows: {result['raw_row_count']}")
    print(f"Episodes per policy: {result['counts_per_policy']}")

    if result["is_clean"]:
        print("\nNo issues found - dataset is fully clean.")
    else:
        print(f"\n{len(result['issues'])} issue(s) found:")
        for issue in result["issues"]:
            print(f"  - {issue}")

    print(f"\nClean rows exported: {result['clean_row_count']} / {result['raw_row_count']}")

    export_path = export_validated_csv(result["df"], args.output)
    print(f"Saved cleaned dataset to: {export_path}")


if __name__ == "__main__":
    main()
