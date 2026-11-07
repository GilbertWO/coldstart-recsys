"""Unit tests for features.py: the leakage guard and the feature values on a tiny example.

    .\\venv\\Scripts\\python.exe -m unittest test_features
"""
import unittest

import numpy as np
import pandas as pd

import features as ft

CUTOFF = pd.Timestamp("2020-09-09")


def history(rows):
    """rows: (customer, article, 'YYYY-MM-DD', price)."""
    return pd.DataFrame(rows, columns=["customer_id", "article_id", "t_dat", "price"]).assign(
        t_dat=lambda d: pd.to_datetime(d["t_dat"]))


def candidates(rows):
    """rows: (customer, article, source, als_score, source_rank)."""
    df = pd.DataFrame(rows, columns=["customer_id", "article_id", "source", "als_score", "source_rank"])
    return df.assign(label=0)


HIST = history([
    ("c1", 10, "2020-09-08", 0.02),  # 1 day before the cutoff: in the 7- and 28-day windows
    ("c1", 10, "2020-08-20", 0.04),  # 20 days before: 28-day window only
    ("c2", 10, "2020-09-02", 0.03),  # 7 days before: in the 7-day window (>= cutoff - 7 days)
    ("c2", 20, "2020-07-01", 0.05),  # 70 days before: in neither window
])
CANDS = candidates([
    ("c1", 10, "als+pop50+recent", 0.9, 1),
    ("c1", 20, "pop50", np.nan, np.nan),
    ("c3", 10, "pop", np.nan, 5),     # customer with no history (cold start)
])


class LeakageGuardTest(unittest.TestCase):
    def test_a_row_on_the_cutoff_day_is_refused(self):
        leaked = pd.concat([HIST, history([("c1", 20, "2020-09-09", 0.01)])], ignore_index=True)
        with self.assertRaises(ValueError):
            ft.build_features(leaked, CANDS, CUTOFF)

    def test_a_row_after_the_cutoff_is_refused(self):
        leaked = pd.concat([HIST, history([("c2", 10, "2020-09-12", 0.01)])], ignore_index=True)
        with self.assertRaises(ValueError):
            ft.build_features(leaked, CANDS, CUTOFF)


class FeatureValuesTest(unittest.TestCase):
    def setUp(self):
        self.f = ft.build_features(HIST, CANDS, CUTOFF).set_index(["customer_id", "article_id"])

    def test_row_order_and_count_are_kept(self):
        out = ft.build_features(HIST, CANDS, CUTOFF)
        self.assertEqual(list(zip(out["customer_id"], out["article_id"])),
                         list(zip(CANDS["customer_id"], CANDS["article_id"])))

    def test_item_popularity_windows(self):
        row = self.f.loc[("c1", 10)]
        self.assertEqual(row["item_cnt_7d"], 2)    # 09-08 and 09-02
        self.assertEqual(row["item_cnt_28d"], 3)   # plus 08-20
        self.assertEqual(self.f.loc[("c1", 20), "item_cnt_28d"], 0)  # only sold 70 days before
        self.assertAlmostEqual(row["item_price_mean_28d"], 0.03, places=6)

    def test_customer_item_history(self):
        self.assertEqual(self.f.loc[("c1", 10), "ci_count"], 2)
        self.assertEqual(self.f.loc[("c1", 10), "ci_days_since"], 1)
        self.assertEqual(self.f.loc[("c1", 20), "ci_count"], 0)
        self.assertTrue(np.isnan(self.f.loc[("c1", 20), "ci_days_since"]))

    def test_customer_history_and_cold_start(self):
        self.assertEqual(self.f.loc[("c1", 10), "cust_n_purchases"], 2)
        self.assertEqual(self.f.loc[("c1", 10), "cust_days_since_last"], 1)
        self.assertEqual(self.f.loc[("c3", 10), "cust_n_purchases"], 0)
        self.assertTrue(np.isnan(self.f.loc[("c3", 10), "cust_days_since_last"]))

    def test_sources_and_als_columns(self):
        als_row, pop_row, cold_row = self.f.loc[("c1", 10)], self.f.loc[("c1", 20)], self.f.loc[("c3", 10)]
        self.assertEqual((als_row["src_als"], als_row["src_pop"], als_row["src_recent"]), (1, 1, 1))
        self.assertEqual((pop_row["src_als"], pop_row["src_pop"]), (0, 1))
        self.assertEqual(als_row["als_rank"], 1)
        self.assertTrue(np.isnan(cold_row["als_rank"]))  # popularity rank is not an ALS rank


if __name__ == "__main__":
    unittest.main()
