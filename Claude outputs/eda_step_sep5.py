"""
Sat Sep 5 deep-block step — coldstart-recsys, Week 1-2.

Run this inside notebooks/eda.ipynb (paste cells in, don't just exec the file).
Prereqs this assumes are already true (check them before running):
  - data/raw/transactions_train.csv exists and is the full file
  - pyarrow is installed (pip install pyarrow)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RAW = Path("data/raw/transactions_train.csv")
CACHE = Path("data/raw/transactions_train.parquet")


def load_transactions(csv_path: Path) -> pd.DataFrame:
    """Memory-efficient full load. On a typical laptop this should stay
    well under a few GB of RAM instead of exploding on default dtypes."""
    dtypes = {
        "customer_id": "category",
        "article_id": "int32",
        "price": "float32",
        "sales_channel_id": "int8",
    }
    df = pd.read_csv(
        csv_path,
        dtype=dtypes,
        parse_dates=["t_dat"],
    )
    return df


# --- load once, cache forever after ---
if CACHE.exists():
    df = pd.read_parquet(CACHE)
else:
    df = load_transactions(RAW)
    df.to_parquet(CACHE)

print(f"Loaded {len(df):,} rows, {df['customer_id'].nunique():,} unique customers")

# --- purchase-frequency distribution per customer ---
purchase_counts = df.groupby("customer_id", observed=True).size()

# --- repeat-purchase rate ---
repeat_rate = (purchase_counts > 1).mean()
print(f"Repeat-purchase rate: {repeat_rate:.2%}")

# --- cold-start ratio: % of customers with fewer than 3 purchases ---
cold_start_ratio = (purchase_counts < 3).mean()
print(f"Cold-start ratio (<3 purchases): {cold_start_ratio:.2%}")

# --- histogram, log-scale y (long-tailed distribution) ---
fig, ax = plt.subplots(figsize=(8, 5))
purchase_counts.plot(kind="hist", bins=50, ax=ax)
ax.set_yscale("log")
ax.set_xlabel("Purchases per customer")
ax.set_ylabel("Number of customers (log scale)")
ax.set_title("Purchase-frequency distribution")
plt.tight_layout()
plt.savefig("notebooks/purchase_frequency_hist.png", dpi=150)
plt.show()
