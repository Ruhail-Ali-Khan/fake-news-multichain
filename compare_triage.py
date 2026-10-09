"""Compare RL triage with a confidence-based policy at exactly the same review budget.

Run after train_model.py and train_rl.py:
    python compare_triage.py

The budget-matched threshold policy is a RETROSPECTIVE evaluation: it uses
unlabeled held-out scores and the number of RL referrals on the test set to
allocate the same count of reviews. Neither test labels nor test correctness
are used to choose reviewed records. This is not a deployed threshold policy.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from train_rl import FAKE, REAL, REVIEW, choose_actions, evaluate

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data" / "processed" / "news_dataset.csv"
MODEL_PATH = BASE / "models" / "fake_news_model.pkl"
POLICY_PATH = BASE / "models" / "rl_policy.npz"
RESULTS_PATH = BASE / "results" / "triage_comparison.json"


def confidence_budget_policy(fake_probabilities, review_count):
    """Review exactly k least-confident samples (closest to 0.5)."""
    probabilities = np.asarray(fake_probabilities, dtype=float)
    if not 0 <= review_count <= len(probabilities):
        raise ValueError("Invalid review count")

    actions = np.where(probabilities >= 0.5, FAKE, REAL).astype(int)
    uncertainty = np.abs(probabilities - 0.5)
    # Stable ordering makes tied confidences reproducible.
    review_indices = np.argsort(uncertainty, kind="stable")[:review_count]
    actions[review_indices] = REVIEW
    return actions


def main():
    for filepath in (DATA_PATH, MODEL_PATH, POLICY_PATH):
        if not filepath.is_file():
            raise FileNotFoundError(f"Missing {filepath}; finish training first.")

    dataset = pd.read_csv(DATA_PATH).dropna(subset=["text_clean", "label"])
    if dataset["text_clean"].duplicated().any():
        raise ValueError("The processed dataset contains duplicate texts")

    X = dataset["text_clean"].astype(str)
    y = dataset["label"].astype(int)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    labels = y_test.to_numpy(dtype=int)

    model = joblib.load(MODEL_PATH)  # Only load your own trusted model file.
    fake_idx = list(model.classes_).index(FAKE)
    fake_prob = model.predict_proba(X_test)[:, fake_idx]

    with np.load(POLICY_PATH, allow_pickle=False) as policy:
        q = policy["q"]
        counts = policy["state_sample_counts"]

    nlp_actions = np.where(fake_prob >= 0.5, FAKE, REAL)
    rl_actions = choose_actions(fake_prob, q, counts)
    review_budget = int(np.sum(rl_actions == REVIEW))
    threshold_actions = confidence_budget_policy(fake_prob, review_budget)

    results = {
        "evaluation_type": "retrospective test-set review-budget matching",
        "notes": [
            "The threshold policy uses the same number of review referrals as RL.",
            "Review selection uses model confidence scores, not test labels.",
            "Neither REVIEW action represents a completed human fact-check.",
            "Only auto-decision accuracy is reported, conditional on not being reviewed.",
        ],
        "review_budget": review_budget,
        "baseline_nlp": evaluate(nlp_actions, labels),
        "rl_policy": evaluate(rl_actions, labels),
        "confidence_review_policy": evaluate(threshold_actions, labels),
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nHELD-OUT TEST: FAIR REVIEW-BUDGET COMPARISON")
    print(f"Articles: {len(labels)}")
    print(f"Matched review referrals: {review_budget}")
    for name, key in (
        ("NLP (no review)", "baseline_nlp"),
        ("RL triage", "rl_policy"),
        ("Confidence-based review", "confidence_review_policy"),
    ):
        result = results[key]
        print(f"\n{name}")
        print("  Reviewed:", result["human_review_count"])
        print("  Incorrect automatic decisions:", result["incorrect_automatic_decisions"])
        print("  Automatic-decision accuracy:",
              f'{result["accuracy_on_automatic_decisions"]:.4%}')
        print("  Mean reward:", f'{result["mean_reward"]:.6f}')

    print(f"\nSaved: {RESULTS_PATH.relative_to(BASE)}")
    print("CAUTION: This compares a one-step RL triage policy, not deep RL.")


if __name__ == "__main__":
    main()
