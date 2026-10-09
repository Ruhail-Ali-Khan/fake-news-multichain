"""Deployable confidence-review threshold vs the one-step Q-learning policy.

Usage:
    python calibrated_triage.py

Threshold is calibrated ONLY on a held-out subset of the original training
partition using an independently fitted NLP pipeline. The original outer
30% test split is never used to choose the threshold or target review rate.

This is a research diagnostic, not a real fact-checking service.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from train_rl import FAKE, REAL, REVIEW, choose_actions, evaluate

BASE = Path(__file__).resolve().parent
DATASET_PATH = BASE / 'data' / 'processed' / 'news_dataset.csv'
MODEL_PATH = BASE / 'models' / 'fake_news_model.pkl'
POLICY_PATH = BASE / 'models' / 'rl_policy.npz'
RESULTS_PATH = BASE / 'results' / 'calibrated_triage.json'

# This is predeclared here, BEFORE evaluating on the outer held-out test set.
TARGET_REVIEW_RATE = 0.10


def action_from_scores(fake_probabilities, threshold):
    probs = np.asarray(fake_probabilities, dtype=float)
    actions = np.where(probs >= 0.5, FAKE, REAL).astype(int)
    actions[np.abs(probs - 0.5) <= threshold] = REVIEW
    return actions


def main():
    for path in (DATASET_PATH, MODEL_PATH, POLICY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f'Missing required file: {path}')

    data = pd.read_csv(DATASET_PATH).dropna(subset=['text_clean', 'label'])
    if data['text_clean'].duplicated().any():
        raise ValueError('Duplicate cleaned texts found')

    X = data['text_clean'].astype(str)
    y = data['label'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    deployed_model = joblib.load(MODEL_PATH)  # Trusted self-trained file only.
    classes = list(deployed_model.classes_)
    fake_col = classes.index(FAKE)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.20, random_state=654,
        stratify=y_train
    )
    print('Training independent calibration model...')
    calibration_model = clone(deployed_model)
    calibration_model.fit(X_fit, y_fit)
    calibration_col = list(calibration_model.classes_).index(FAKE)
    calibration_scores = calibration_model.predict_proba(X_cal)[:, calibration_col]
    uncertainties = np.abs(calibration_scores - 0.5)

    # Rank a fixed fraction from calibration only. This threshold is then
    # frozen and applied without adjustment to the previously unseen test set.
    count = max(1, int(round(TARGET_REVIEW_RATE * len(uncertainties))))
    cutoff = float(np.sort(uncertainties)[count - 1])

    test_scores = deployed_model.predict_proba(X_test)[:, fake_col]
    labels = y_test.to_numpy(dtype=int)
    base_actions = np.where(test_scores >= 0.5, FAKE, REAL)

    with np.load(POLICY_PATH, allow_pickle=False) as stored:
        q = stored['q']
        counts = stored['state_sample_counts']
    rl_actions = choose_actions(test_scores, q, counts)
    threshold_actions = action_from_scores(test_scores, cutoff)

    results = {
        'method': 'Training-only calibrated fixed-10%-target review policy',
        'target_review_rate': TARGET_REVIEW_RATE,
        'actual_calibration_referral_rate': float(np.mean(uncertainties <= cutoff)),
        'frozen_confidence_margin': cutoff,
        'calibration_sample_count': int(len(X_cal)),
        'test_sample_count': int(len(X_test)),
        'cautions': [
            'The threshold was selected using training-partition calibration scores, not test scores or test labels.',
            'The review rates for RL and the calibrated policy need not match on test data.',
            'Referral for review is simulated; no human reviewed these articles.',
            'Scores are not calibrated real-world truth probabilities.',
            'Q-learning is one-step tabular RL, not the paper\'s deep RL.',
        ],
        'baseline_nlp': evaluate(base_actions, labels),
        'qlearning_triage': evaluate(rl_actions, labels),
        'calibrated_confidence_triage': evaluate(threshold_actions, labels),
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding='utf-8')

    print('\nTRAINING-ONLY CALIBRATED REVIEW POLICY')
    print('Target review rate:', f'{TARGET_REVIEW_RATE:.1%}')
    print('Calibration articles:', len(X_cal))
    print('Frozen uncertainty threshold:', f'{cutoff:.6f}')
    print('Held-out test articles:', len(X_test))

    for name, key in [
        ('NLP baseline', 'baseline_nlp'),
        ('Q-learning triage', 'qlearning_triage'),
        ('Calibrated confidence triage', 'calibrated_confidence_triage')
    ]:
        value = results[key]
        print(f'\n{name}')
        print('  Review referrals:', value['human_review_count'])
        print('  Review rate:', f"{value['human_review_rate']:.2%}")
        print('  Incorrect automatic decisions:', value['incorrect_automatic_decisions'])
        print('  Automatic-decision accuracy:', f"{value['accuracy_on_automatic_decisions']:.4%}")
        print('  Mean reward:', f"{value['mean_reward']:.6f}")

    print('\nSaved:', RESULTS_PATH.relative_to(BASE))
    print('NOTE: Review is simulated; RL and calibrated policy may refer different numbers of articles.')


if __name__ == '__main__':
    main()
