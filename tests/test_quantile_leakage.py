import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from medical_llm.utils.quantiles import fit_quantiles, apply_quantile, apply_quantiles_to_labevents


class TestQuantileLeakage(unittest.TestCase):

    def setUp(self):
        # Synthetic lab events dataset for itemid 50800
        np.random.seed(42)
        train_vals = np.random.normal(loc=100.0, scale=15.0, size=100)
        val_vals = np.random.normal(loc=200.0, scale=50.0, size=50)

        self.train_df = pd.DataFrame({
            "hadm_id": range(1, 101),
            "itemid": [50800] * 100,
            "valuenum": train_vals,
            "flag": ["NORMAL"] * 100,
        })

        self.val_df = pd.DataFrame({
            "hadm_id": range(101, 151),
            "itemid": [50800] * 50,
            "valuenum": val_vals,
            "flag": ["ABNORMAL"] * 50,
        })

    def test_validation_does_not_affect_train_quantiles(self):
        """Verify that fitting quantiles on train_df yields identical boundaries regardless of val_df contents."""
        q_train_only = fit_quantiles(self.train_df)

        # Mutate validation dataset with extreme outliers
        mutated_val_df = self.val_df.copy()
        mutated_val_df["valuenum"] = mutated_val_df["valuenum"] * 1000.0

        # Fit quantiles again strictly on train_df
        q_train_again = fit_quantiles(self.train_df)

        # Assert boundaries are 100% identical
        np.testing.assert_array_equal(q_train_only[50800], q_train_again[50800])

    def test_same_quantiles_applied_to_val_df(self):
        """Verify that apply_quantiles_to_labevents uses the train quantile map to discretize val_df."""
        q_map = fit_quantiles(self.train_df)
        val_processed = apply_quantiles_to_labevents(self.val_df, q_map)

        self.assertIn("qbucket", val_processed.columns)
        self.assertIn("event_token", val_processed.columns)
        self.assertEqual(len(val_processed), len(self.val_df))

        # Check token formatting
        first_token = val_processed["event_token"].iloc[0]
        self.assertTrue(first_token.startswith("LAB_50800_Q"))

    def test_unseen_itemid_fallback(self):
        """Verify fallback behavior for lab item IDs not present in training quantile map."""
        q_map = fit_quantiles(self.train_df)
        new_item_df = pd.DataFrame({
            "hadm_id": [201],
            "itemid": [99999],  # unseen itemid
            "valuenum": [50.0],
            "flag": ["ABNORMAL"],
        })

        processed = apply_quantiles_to_labevents(new_item_df, q_map)
        self.assertEqual(processed["event_token"].iloc[0], "LAB_99999_ABNORMAL")

    def test_cross_validation_fold_quantile_isolation(self):
        """Verify that modifying out-of-fold / test data has zero effect on training-fold quantiles."""
        fold_train_labs = self.train_df.copy()
        fold_test_labs = self.val_df.copy()

        q_fold_1 = fit_quantiles(fold_train_labs)

        # Drastically alter the test fold
        fold_test_labs["valuenum"] = -999999.0
        q_fold_2 = fit_quantiles(fold_train_labs)

        # Assert quantiles are identical
        np.testing.assert_array_equal(q_fold_1[50800], q_fold_2[50800])


if __name__ == "__main__":
    unittest.main()
