"""Train a one-step Q-learning (contextual bandit) news triage policy.

This is a paper-inspired RL baseline, NOT a reproduction of the paper's deep RL.
Labels are used as simulated reviewer feedback during training only.
The original 30% held-out test partition is never used to fit the policy.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

BASE = Path(__file__).resolve().parent
DATASET = BASE / "data" / "processed" / "news_dataset.csv"
MODEL = BASE / "models" / "fake_news_model.pkl"
POLICY = BASE / "models" / "rl_policy.npz"
RESULTS = BASE / "results"

# Actions: 0 = flag FAKE, 1 = accept REAL, 2 = request human REVIEW.
FAKE, REAL, REVIEW = 0, 1, 2
ACTION_NAMES = ["FAKE", "REAL", "REVIEW"]
NUM_BINS = 20
MIN_STATE_SAMPLES = 25
REWARD_CORRECT = 1.0
PENALTY_WRONG = -8.0
COST_REVIEW = -0.25


def bucketize(fake_probabilities):
    probabilities = np.asarray(fake_probabilities, dtype=float)
    return np.clip((probabilities * NUM_BINS).astype(int), 0, NUM_BINS - 1)


def reward(action, actual_label):
    if action == REVIEW:
        return COST_REVIEW
    return REWARD_CORRECT if action == actual_label else PENALTY_WRONG


def train_qlearning(fake_probabilities, labels, training_indices):
    """One-step episodes: choose action, get reward, terminate."""
    rng = np.random.default_rng(42)
    states = bucketize(fake_probabilities)
    q = np.zeros((NUM_BINS, 3), dtype=float)
    # Unique training examples per bucket, rather than repeated RL visits.
    state_sample_counts = np.bincount(
        states[training_indices], minlength=NUM_BINS
    )

    # A contextual bandit is a special case of an MDP with terminal next state.
    # For terminal episodes: Q(s,a) <- Q(s,a) + alpha * [r - Q(s,a)].
    for epoch in range(30):
        epsilon = max(0.05, 0.6 * (0.9 ** epoch))
        for index in rng.permutation(training_indices):
            state = int(states[index])
            action = (
                int(rng.integers(3)) if rng.random() < epsilon
                else int(np.argmax(q[state]))
            )
            observed_reward = reward(action, int(labels[index]))
            q[state, action] += 0.08 * (observed_reward - q[state, action])
    return q, state_sample_counts


def choose_actions(fake_probabilities, q, state_sample_counts):
    states = bucketize(fake_probabilities)
    actions = np.argmax(q[states], axis=1).astype(int)
    # An under-observed state cannot justify an automatic decision.
    actions[state_sample_counts[states] < MIN_STATE_SAMPLES] = REVIEW
    return actions


def evaluate(actions, labels):
    labels = np.asarray(labels, dtype=int)
    actions = np.asarray(actions, dtype=int)
    reviewed = actions == REVIEW
    automated = ~reviewed
    correct = automated & (actions == labels)
    wrong = automated & (actions != labels)
    individual_rewards = np.where(
        reviewed, COST_REVIEW,
        np.where(correct, REWARD_CORRECT, PENALTY_WRONG)
    )
    return {
        "articles": int(len(labels)),
        "human_review_count": int(reviewed.sum()),
        "human_review_rate": float(reviewed.mean()),
        "automated_count": int(automated.sum()),
        "correct_automatic_decisions": int(correct.sum()),
        "incorrect_automatic_decisions": int(wrong.sum()),
        "accuracy_on_automatic_decisions": (
            float(correct.sum() / automated.sum()) if automated.any() else None
        ),
        "mean_reward": float(individual_rewards.mean()),
        "action_counts": {
            ACTION_NAMES[action]: int((actions == action).sum())
            for action in range(3)
        },
    }


def main():
    data = pd.read_csv(DATASET).dropna(subset=["text_clean", "label"])
    if data["text_clean"].duplicated().any():
        raise ValueError("Duplicate cleaned articles exist; check preprocessing.")
    X, y = data["text_clean"].astype(str), data["label"].astype(int)

    # EXACTLY the same outer 70/30 split as the existing train_model.py.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    model = joblib.load(MODEL)  # Load only your own trusted saved model.
    fake_column = list(model.classes_).index(FAKE)

    # Avoid in-sample model scores for RL: create out-of-fold probabilities
    # from clones fitted on training folds only. The test set remains untouched.
    print("Creating 3-fold out-of-fold scores for RL; this retrains the NLP model 3 times.", flush=True)
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    out_of_fold = cross_val_predict(
        model, X_train, y_train, cv=folds,
        method="predict_proba", n_jobs=1
    )[:, fake_column]

    labels_train = y_train.to_numpy(dtype=int)
    all_indices = np.arange(len(labels_train))
    train_idx, val_idx = train_test_split(
        all_indices, test_size=0.20, random_state=123,
        stratify=labels_train
    )

    q, state_sample_counts = train_qlearning(out_of_fold, labels_train, train_idx)
    validation_actions = choose_actions(out_of_fold[val_idx], q, state_sample_counts)
    validation_metrics = evaluate(validation_actions, labels_train[val_idx])

    # Final held-out evaluation uses the *original fitted* model on X_test.
    test_probabilities = model.predict_proba(X_test)[:, fake_column]
    labels_test = y_test.to_numpy(dtype=int)
    baseline_actions = np.where(test_probabilities >= 0.5, FAKE, REAL)
    rl_actions = choose_actions(test_probabilities, q, state_sample_counts)

    baseline_metrics = evaluate(baseline_actions, labels_test)
    rl_metrics = evaluate(rl_actions, labels_test)

    RESULTS.mkdir(exist_ok=True)
    POLICY.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        POLICY, q=q, state_sample_counts=state_sample_counts,
        num_bins=np.array(NUM_BINS),
        min_state_samples=np.array(MIN_STATE_SAMPLES)
    )
    metrics = {
        "method": "one-step tabular Q-learning contextual-bandit baseline",
        "note": "Not the paper's deep reinforcement learning architecture.",
        "reward_correct": REWARD_CORRECT,
        "penalty_wrong": PENALTY_WRONG,
        "cost_human_review": COST_REVIEW,
        "training_samples_for_rl": int(len(train_idx)),
        "validation_samples_for_rl": int(len(val_idx)),
        "held_out_test_samples": int(len(labels_test)),
        "validation": validation_metrics,
        "baseline_held_out_test": baseline_metrics,
        "rl_held_out_test": rl_metrics,
    }
    (RESULTS / "rl_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Useful for the report: map each fake-probability interval to the RL action.
    action_per_bucket = choose_actions(
        (np.arange(NUM_BINS) + 0.5) / NUM_BINS, q, state_sample_counts
    )
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.step(
        (np.arange(NUM_BINS) + 0.5) / NUM_BINS,
        action_per_bucket, where="mid"
    )
    ax.set_yticks([FAKE, REAL, REVIEW], ACTION_NAMES)
    ax.set_xlabel("Fake class score from NLP baseline")
    ax.set_ylabel("RL action")
    ax.set_title("Learned triage policy by score bucket")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(RESULTS / "rl_policy_actions.png", dpi=200)
    plt.close(fig)

    print("\nRL VALIDATION (out-of-fold probabilities)")
    print(json.dumps(validation_metrics, indent=2))
    print("\nORIGINAL NLP BASELINE — HELD-OUT TEST")
    print(json.dumps(baseline_metrics, indent=2))
    print("\nRL TRIAGE POLICY — SAME HELD-OUT TEST")
    print(json.dumps(rl_metrics, indent=2))
    print("\nSaved: models/rl_policy.npz, results/rl_metrics.json, results/rl_policy_actions.png")
    print("NOTE: Human review is simulated as a referral, not an actual human verification.")


if __name__ == "__main__":
    main()
