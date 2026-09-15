"""
prepare_data.py — Week 3 data pipeline for coldstart-recsys.

Produces clean, temporally-split train/val/test parquet files from the raw
H&M Personalized Fashion Recommendations CSVs.

DESIGN DECISION — why a global date cutoff, not a per-customer cutoff:
    The real Kaggle competition evaluates every submission on ONE held-out
    future week, scored across the whole customer base (MAP@12), not on a
    per-customer leave-one-out basis. Splitting globally by date reproduces
    that exact structure, so offline val/test numbers computed here are
    directly comparable to what "the real test" would have measured, and
    the eval story is trivial to explain in an interview: "I held out the
    last two weeks of the training window, the same way Kaggle held out
    the week after it." A per-customer time-split (each customer's own last
    N transactions in test) is a legitimate design for some recsys papers,
    but it answers a different question ("can we predict this user's next
    purchase given their own history") and would make val/test metrics
    incomparable to the competition's own leaderboard framing. Don't mix
    the two.

DESIGN DECISION — cutoff granularity:
    Held-out set = last 1 full week of transactions_train's date range.
    Validation set = the 1 full week immediately before that.
    Everything earlier = train.
    Rationale: recsys models here need long purchase history to build
    co-purchase / popularity features; a single week is already enough
    data to compute MAP@12-style ranking metrics on, and it mirrors the
    competition's own one-week-ahead test format. Change N_VAL_WEEKS /
    N_TEST_WEEKS below if your own EDA (check the actual min/max t_dat
    span first) suggests the tail is too sparse for 1 week to be stable.

LEAK WARNING beyond row-splitting (the plan text doesn't call this out,
but it's the second most common way this kind of project silently leaks):
    Any feature computed from full-history stats (item popularity, a
    customer's total purchase count, repeat-purchase rate, etc.) MUST be
    computed using only data available as of that row's split cutoff —
    never from the full dataset. This script does not compute modeling
    features (that's Week 4+), but if/when you build a feature step on
    top of these parquet files, recompute popularity/aggregate features
    per-split from data strictly before that split's cutoff date. Fitting
    a "global popularity" feature on the full dataset before splitting is
    the second-most-common leak in projects like this, after a bad split.

Usage:
    python prepare_data.py --data-dir data/raw --out-dir data/processed
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- tunables -----------------------------------------------------------
N_TEST_WEEKS = 1   # most recent full week(s) -> test
N_VAL_WEEKS = 1     # full week(s) immediately before test -> val
# --------------------------------------------------------------------------


def load_transactions(data_dir: Path) -> pd.DataFrame:
    """Load the cached parquet if present (Week 1-2 output); fall back to CSV."""
    parquet_path = data_dir / "transactions_train.parquet"
    if parquet_path.exists():
        log.info("Loading cached parquet: %s", parquet_path)
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = data_dir / "transactions_train.csv"
        log.info("No cached parquet found, loading CSV (slower): %s", csv_path)
        df = pd.read_csv(
            csv_path,
            dtype={
                "customer_id": "category",
                "article_id": "int32",
                "price": "float32",
                "sales_channel_id": "int8",
            },
            parse_dates=["t_dat"],
        )
    if not pd.api.types.is_datetime64_any_dtype(df["t_dat"]):
        df["t_dat"] = pd.to_datetime(df["t_dat"])
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    DECISION (reversed from an earlier version of this script -- see repo
    history / conversation for why): exact-duplicate rows (same customer,
    article, date, price, channel) are NOT dropped. This dataset has no
    quantity column, so a customer buying 2 units of the same article on
    the same day is logged as two identical rows. Verified against this
    repo's own eda.ipynb: cell 7 computes purchase_counts via
    transactions.groupby('customer_id').size() on the undeduplicated data,
    and README.md's published findings (19.58% cold-start ratio, 89.71%
    repeat-purchase rate, 31,788,324 total transactions) are all built on
    that same undeduplicated count. Dropping duplicates here would make
    this pipeline's transaction count disagree with the numbers already
    published in your own README -- so don't, unless you deliberately
    redo the EDA numbers to match.

    Still logged (not dropped) so the count is visible without acting on it.
    """
    before = len(df)

    exact_dupes = df.duplicated().sum()
    if exact_dupes:
        log.info(
            "%d / %d rows (%.3f%%) are exact duplicates (same customer/article/date/price/"
            "channel) -- kept, not dropped, since they're plausibly repeat same-day unit "
            "purchases and your published EDA numbers already count them as real transactions.",
            exact_dupes, before, 100 * exact_dupes / before,
        )

    required = ["customer_id", "article_id", "t_dat"]
    df = df.dropna(subset=required)
    na_dropped = before - len(df)

    n1 = len(df)
    df = df[df["price"] > 0]
    price_dropped = n1 - len(df)

    total_dropped = before - len(df)
    log.info(
        "Cleaning breakdown -- missing key fields: %d (%.3f%%), non-positive price: %d (%.3f%%), "
        "TOTAL DROPPED: %d / %d (%.3f%%)  [duplicates retained, see above]",
        na_dropped, 100 * na_dropped / before,
        price_dropped, 100 * price_dropped / before,
        total_dropped, before, 100 * total_dropped / before,
    )
    return df


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="customer_id")

    # FN / Active are 1.0-or-NaN flag columns in this dataset -- NaN means
    # "flag not set", i.e. false, not "unknown". Fill accordingly rather
    # than dropping ~65% of customers, which is what dropna() would do here.
    for col in ("FN", "Active"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype("int8")

    if "club_member_status" in df.columns:
        df["club_member_status"] = df["club_member_status"].fillna("UNKNOWN")

    if "fashion_news_frequency" in df.columns:
        df["fashion_news_frequency"] = (
            df["fashion_news_frequency"].fillna("NONE").replace("None", "NONE")
        )

    # age has a small missing rate (check the exact number from your own EDA
    # notebook -- README should already state it). Median-impute and keep an
    # explicit flag so a model can still distinguish "imputed" from "known".
    if "age" in df.columns:
        df["age_missing"] = df["age"].isna().astype("int8")
        df["age"] = df["age"].fillna(df["age"].median()).astype("float32")

    return df


def clean_articles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="article_id")
    # Free-text / near-unique-cardinality columns identified in EDA finding #3
    # (detail_desc, prod_name, product_code, detail_desc) aren't modeling
    # features -- drop them here so every downstream consumer of this parquet
    # gets the same, already-decided column set instead of re-deciding it.
    drop_cols = [c for c in ("detail_desc", "prod_name", "product_code") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def temporal_split(df: pd.DataFrame, n_val_weeks: int, n_test_weeks: int):
    """Global date-based split. See module docstring for why global, not per-customer."""
    max_date = df["t_dat"].max()
    test_start = max_date.normalize() - pd.Timedelta(weeks=n_test_weeks) + pd.Timedelta(days=1)
    val_start = test_start - pd.Timedelta(weeks=n_val_weeks)

    train = df[df["t_dat"] < val_start]
    val = df[(df["t_dat"] >= val_start) & (df["t_dat"] < test_start)]
    test = df[df["t_dat"] >= test_start]

    log.info("Split cutoffs -- val_start=%s, test_start=%s, max_date=%s",
              val_start.date(), test_start.date(), max_date.date())
    for name, part in (("train", train), ("val", val), ("test", test)):
        log.info("  %-5s: %10d rows | %s -> %s", name, len(part),
                  part["t_dat"].min(), part["t_dat"].max())

    if len(val) == 0 or len(test) == 0:
        raise ValueError(
            "val or test split is empty -- check that your data actually spans "
            "at least (n_val_weeks + n_test_weeks) full weeks before relying on this split."
        )
    return train, val, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-weeks", type=int, default=N_VAL_WEEKS)
    parser.add_argument("--test-weeks", type=int, default=N_TEST_WEEKS)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw tables...")
    transactions = load_transactions(args.data_dir)
    customers = pd.read_csv(args.data_dir / "customers.csv")
    articles = pd.read_csv(args.data_dir / "articles.csv")

    log.info("Cleaning...")
    transactions = clean_transactions(transactions)
    customers = clean_customers(customers)
    articles = clean_articles(articles)

    log.info("Splitting transactions by time (global date cutoff)...")
    train, val, test = temporal_split(transactions, args.val_weeks, args.test_weeks)

    log.info("Writing parquet outputs to %s ...", args.out_dir)
    train.to_parquet(args.out_dir / "transactions_train_split.parquet", index=False)
    val.to_parquet(args.out_dir / "transactions_val.parquet", index=False)
    test.to_parquet(args.out_dir / "transactions_test.parquet", index=False)
    customers.to_parquet(args.out_dir / "customers_clean.parquet", index=False)
    articles.to_parquet(args.out_dir / "articles_clean.parquet", index=False)

    log.info("Done.")


if __name__ == "__main__":
    main()
