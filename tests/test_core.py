"""Pure-function regression checks; no real RPC server or training dataset required."""
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

# Import news_pipeline without requiring a real Windows MultiChain installation.
fake_client = types.ModuleType("multichain_client")
fake_client.STREAM_NAME = "news_records"
fake_client.rpc_call = lambda *args, **kwargs: None
with patch.dict(sys.modules, {"multichain_client": fake_client}):
    import news_pipeline

from train_rl import FAKE, REAL, REVIEW, choose_actions, evaluate, reward, NUM_BINS


class HashAndPolicyTests(unittest.TestCase):
    def test_full_article_hash_is_stable(self):
        self.assertEqual(
            news_pipeline.article_hash("headline", "content"),
            news_pipeline.article_hash("headline", "content"),
        )

    def test_full_article_hash_detects_headline_change(self):
        self.assertNotEqual(
            news_pipeline.article_hash("headline", "content"),
            news_pipeline.article_hash("changed headline", "content"),
        )

    def test_full_article_hash_detects_content_change(self):
        self.assertNotEqual(
            news_pipeline.article_hash("headline", "content"),
            news_pipeline.article_hash("headline", "changed content"),
        )

    def test_qlearning_reward_rules(self):
        self.assertEqual(reward(FAKE, FAKE), 1.0)
        self.assertEqual(reward(REAL, FAKE), -8.0)
        self.assertEqual(reward(REVIEW, FAKE), -0.25)

    def test_qlearning_forces_review_for_unseen_bin(self):
        q = np.zeros((NUM_BINS, 3))
        q[:, REAL] = 1.0
        count = np.ones(NUM_BINS, dtype=int) * 100
        count[10] = 0
        self.assertEqual(int(choose_actions([0.51], q, count)[0]), REVIEW)

    def test_evaluation_counts(self):
        stats = evaluate([FAKE, REAL, REVIEW], [FAKE, FAKE, REAL])
        self.assertEqual(stats["automated_count"], 2)
        self.assertEqual(stats["incorrect_automatic_decisions"], 1)
        self.assertEqual(stats["human_review_count"], 1)
        self.assertAlmostEqual(stats["mean_reward"], (1 - 8 - 0.25) / 3)


if __name__ == "__main__":
    unittest.main()
