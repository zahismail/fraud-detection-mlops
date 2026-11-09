
from pathlib import Path
import argparse

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


# Project configuration
BASE_DIR = Path(__file__).resolve().parent
FEATURES = ["TX_AMOUNT", "TX_HOUR", "TX_DAY_OF_WEEK"]

REFERENCE_PATH = BASE_DIR / "processed" / "train.pkl"
CURRENT_PATH = BASE_DIR / "processed" / "test.pkl"
OUTPUT_DIR = BASE_DIR / "monitoring"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Simulate a change in transaction amounts"
    )
    args = parser.parse_args()

    # 1. Load datasets
    reference_df = pd.read_pickle(REFERENCE_PATH)
    current_df = pd.read_pickle(CURRENT_PATH)

    # 2. Select the features used by our model
    reference = reference_df[FEATURES].sample(
        n=min(10000, len(reference_df)),
        random_state=42
    ).copy()

    current = current_df[FEATURES].sample(
        n=min(10000, len(current_df)),
        random_state=42
    ).copy()

    # 3. Optionally simulate data drift
    if args.simulate_drift:
        current["TX_AMOUNT"] = current["TX_AMOUNT"] * 2

    # 4. Generate the drift report
    report = Report([
        DataDriftPreset()
    ])

    result = report.run(
        current_data=current,
        reference_data=reference
    )

    # 5. Save the HTML report
    OUTPUT_DIR.mkdir(exist_ok=True)

    filename = (
        "drift_simulated.html"
        if args.simulate_drift
        else "drift_baseline.html"
    )

    output_path = OUTPUT_DIR / filename
    result.save_html(str(output_path))

    print(f"Monitoring report generated: {output_path}")


if __name__ == "__main__":
    main()
