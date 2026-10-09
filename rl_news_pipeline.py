"""Publish an NLP prediction and one-step Q-learning triage recommendation to MultiChain.

Run inside the existing fake-news-blockchain project:
    python rl_news_pipeline.py

This is a research demonstration. Neither the model prediction nor the RL action
is independent fact verification. Never treat an action as a verified verdict.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from multichain_client import STREAM_NAME, rpc_call
from news_pipeline import article_hash, sha256_text
from predict_news import load_model
from prepare_dataset import clean_text
from train_rl import ACTION_NAMES, MIN_STATE_SAMPLES, NUM_BINS, choose_actions

BASE_DIR = Path(__file__).resolve().parent
POLICY_FILE = BASE_DIR / "models" / "rl_policy.npz"
DATA_DIR = BASE_DIR / "data"


def load_policy():
    if not POLICY_FILE.is_file():
        raise FileNotFoundError("Run python train_rl.py before using this script.")

    with np.load(POLICY_FILE, allow_pickle=False) as policy:
        q = policy["q"]
        counts = policy["state_sample_counts"]
        num_bins = int(policy["num_bins"])
        min_samples = int(policy["min_state_samples"])

    if (
        num_bins != NUM_BINS
        or min_samples != MIN_STATE_SAMPLES
        or q.shape != (NUM_BINS, 3)
        or counts.shape != (NUM_BINS,)
    ):
        raise ValueError("RL policy format differs from train_rl.py; retrain the policy.")
    return q, counts


def submit_news():
    print("\nNLP + RL + MULTICHAIN NEWS DEMO")
    print("--------------------------------")
    headline = input("Enter headline: ").strip()
    content = input("Enter full article text on one line: ").strip()
    if not headline or not content:
        raise ValueError("Both headline and article content are required.")

    processed = clean_text(headline + " " + content)
    if len(processed.split()) < 25:
        raise ValueError("Please enter at least 25 words of article text.")

    model = load_model()
    q, counts = load_policy()

    class_scores = model.predict_proba([processed])[0]
    probabilities = {
        int(label): float(score)
        for label, score in zip(model.classes_, class_scores)
    }
    if set(probabilities) != {0, 1}:
        raise ValueError("Unexpected model classes. Expected 0=Fake, 1=Real.")

    fake_probability = probabilities[0]
    predicted_label = "FAKE" if fake_probability >= 0.5 else "REAL"
    rl_action = int(choose_actions(
        np.array([fake_probability]), q, counts
    )[0])
    action_name = ACTION_NAMES[rl_action]

    article_id = "news_" + uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()
    article = {
        "article_id": article_id,
        "headline": headline,
        "content": content,
        "created_at": timestamp,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    article_path = DATA_DIR / f"{article_id}.json"
    article_path.write_text(
        json.dumps(article, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    blockchain_record = {
        "article_id": article_id,
        "headline": headline,
        "prediction": predicted_label,
        "rl_action": action_name,
        "review_required_by_policy": action_name == "REVIEW",
        "fake_score": round(fake_probability, 6),
        "real_score": round(probabilities[1], 6),
        "model": "TFIDF_LogisticRegression_v1",
        "policy": "one_step_tabular_Q_learning_v1",
        "content_sha256": sha256_text(content),
        "article_sha256": article_hash(headline, content),
        "hash_scheme": "SHA256_canonical_JSON_v1",
        "submitted_at": timestamp,
        "status": "pending_review",
    }

    txid = rpc_call(
        "publish", [STREAM_NAME, article_id, {"json": blockchain_record}]
    )

    print("\nRECORD SUBMITTED TO MULTICHAIN")
    print("Article ID:        ", article_id)
    print("NLP prediction:    ", predicted_label)
    print(f"NLP fake score:     {fake_probability:.4f}")
    print("RL recommendation: ", action_name)
    print("Verification status: PENDING — no human verification performed")
    print("Blockchain txid:   ", txid)
    print("Local article file:", article_path)
    print("\nVerify integrity after confirmation:")
    print(f"python news_pipeline.py --verify {article_id}")
    print("\nWARNING: These labels are predictions, not verified facts.")


if __name__ == "__main__":
    try:
        submit_news()
    except Exception as exc:
        print("ERROR:", exc)
        raise SystemExit(1)
