from pathlib import Path
import pandas as pd

# 1. Load DVC-tracked dataset
data_dir = Path(__file__).resolve().parent / "data"
files = sorted(data_dir.glob("*.pkl"))

if not files:
    raise FileNotFoundError("No .pkl files found in data/")

df = pd.concat(
    [pd.read_pickle(file) for file in files],
    ignore_index=True
)

# 2. Sort chronologically
df["TX_DATETIME"] = pd.to_datetime(df["TX_DATETIME"])
df = df.sort_values("TX_DATETIME").reset_index(drop=True)

# 3. Create features
df["TX_HOUR"] = df["TX_DATETIME"].dt.hour
df["CUSTOMER_AVG_AMOUNT"] = (
    df.groupby("CUSTOMER_ID")["TX_AMOUNT"]
      .transform(lambda x: x.shift(1).expanding().mean())
)
df["CUSTOMER_TX_COUNT"] = (
    df.groupby("CUSTOMER_ID").cumcount()
)
df["AMOUNT_TO_AVG_RATIO"] = (
    df["TX_AMOUNT"] /
    df["CUSTOMER_AVG_AMOUNT"].clip(lower=1)
)
df["TX_DAY_OF_WEEK"] = df["TX_DATETIME"].dt.day_of_week

# 4. Split data
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# 5. Save prepared datasets
output_dir = Path("processed")
output_dir.mkdir(exist_ok=True)

train_df.to_pickle(output_dir / "train.pkl")
val_df.to_pickle(output_dir / "val.pkl")
test_df.to_pickle(output_dir / "test.pkl")

print("Data preparation completed!")
print(f"Train: {len(train_df)} transactions")
print(f"Validation: {len(val_df)} transactions")
print(f"Test: {len(test_df)} transactions")