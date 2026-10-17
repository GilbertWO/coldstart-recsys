import datetime as dt
import json
import unittest
from pathlib import Path

import pandas as pd

import refresh_fallback as rf

D = dt.date


def frame(rows):
    """rows: (article_id, 'YYYY-MM-DD') pairs -> a minimal transactions frame."""
    return pd.DataFrame(
        {
            "customer_id": [f"c{i}" for i in range(len(rows))],
            "article_id": [a for a, _ in rows],
            "t_dat": pd.to_datetime([d for _, d in rows]),
        }
    )


class RecentTopTest(unittest.TestCase):
    def test_window_is_the_seven_days_ending_at_as_of(self):
        tx = frame([
            (1, "2020-09-15"),  # as_of day: in
            (2, "2020-09-09"),  # six days before: in
            (3, "2020-09-08"),  # exactly seven days before: out
            (4, "2020-09-16"),  # after as_of: out
        ])
        self.assertEqual(sorted(rf.recent_top(tx, D(2020, 9, 15), days=7, k=10)), [1, 2])

    def test_window_is_anchored_to_as_of_not_to_the_last_sale(self):
        tx = frame([
            (7, "2020-09-06"), (7, "2020-09-06"),  # old: out for as_of 09-15
            (8, "2020-09-10"),                      # in
        ])
        # No sales on 09-13..09-15. A window anchored to the last sale (09-10) would wrongly include article 7.
        self.assertEqual(rf.recent_top(tx, D(2020, 9, 15), days=7, k=10), [8])

    def test_ties_break_by_article_id(self):
        tx = frame([(5, "2020-09-14"), (3, "2020-09-14"), (9, "2020-09-14"), (9, "2020-09-13")])
        self.assertEqual(rf.recent_top(tx, D(2020, 9, 15), days=7, k=3), [9, 3, 5])


class ValidateItemsTest(unittest.TestCase):
    def test_accepts_twelve_distinct_ids(self):
        self.assertEqual(rf.validate_items(list(range(1, 13))), list(range(1, 13)))

    def test_rejects_short_or_duplicated_lists(self):
        with self.assertRaises(ValueError):
            rf.validate_items([1, 2, 3])
        with self.assertRaises(ValueError):
            rf.validate_items([1] * 12)


class CheckAsOfTest(unittest.TestCase):
    DATA_MIN, DATA_MAX, LATEST = D(2018, 9, 20), D(2020, 9, 22), D(2020, 9, 15)

    def check(self, as_of):
        rf.check_as_of(as_of, self.DATA_MIN, self.DATA_MAX, self.LATEST)

    def test_accepts_a_later_date_and_a_rerun_of_the_current_one(self):
        self.check(D(2020, 9, 22))
        self.check(D(2020, 9, 15))

    def test_refuses_dates_the_api_would_ignore_or_the_data_cannot_support(self):
        for bad in (D(2020, 9, 14), D(2020, 9, 23), D(2018, 9, 25)):
            with self.subTest(as_of=bad), self.assertRaises(ValueError):
                self.check(bad)


class BuildInsertSqlTest(unittest.TestCase):
    def test_is_one_idempotent_insert_that_carries_the_model_version_over(self):
        sql = rf.build_insert_sql(D(2020, 9, 22), 7, list(range(1, 13)))
        self.assertIn("DATE '2020-09-22'", sql)
        self.assertIn("ARRAY[1,2,3,4,5,6,7,8,9,10,11,12]", sql)
        self.assertIn("ON CONFLICT (source, as_of) DO UPDATE", sql)
        self.assertIn("FROM fallback_lists", sql)
        self.assertIn("model_version", sql)


class ParityTest(unittest.TestCase):
    def test_history_end_reproduces_the_stored_fallback_list(self):
        root = Path(__file__).parent
        meta_path = root / "models" / "als_meta.json"
        parts = [root / "data" / "processed" / n for n in
                 ("transactions_train_split.parquet", "transactions_val.parquet")]
        if not meta_path.exists() or not all(p.exists() for p in parts):
            self.skipTest("model artifact or processed data not available")
        meta = json.loads(meta_path.read_text())
        history = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        as_of = dt.date.fromisoformat(meta["history_end"])
        expected = [int(a) for a in meta["fallback_top12"]]
        self.assertEqual(rf.recent_top(history, as_of, days=7, k=12), expected)


if __name__ == "__main__":
    unittest.main()
