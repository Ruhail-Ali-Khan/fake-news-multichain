"""Evaluate the frozen ISOT-trained model on the out-of-domain LIAR benchmark.

1. Download original LIAR dataset (Wang, ACL 2017) separately.
2. Extract test.tsv to data/external/liar/test.tsv.
3. Run: python evaluate_liar_external.py

This evaluates *short political statements*, NOT full news articles.
The test set must never be used to tune the model or decision threshold.
Only unambiguous labels are included:
    false, pants-fire -> FAKE (0)
    true              -> REAL (1)
    barely-true, half-true, mostly-true -> excluded
"""

import csv
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from prepare_dataset import clean_text

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data' / 'external' / 'liar' / 'test.tsv'
MODEL = BASE / 'models' / 'fake_news_model.pkl'
RESULTS = BASE / 'results' / 'liar_external_evaluation.json'
LABEL_MAP = {'false': 0, 'pants-fire': 0, 'true': 1}


def main():
    if not DATA.is_file():
        raise FileNotFoundError(
            f'LIAR test dataset not found: {DATA}\n'
            'Download liar_dataset.zip from the original LIAR project, '
            'then extract test.tsv to that location.'
        )
    if not MODEL.is_file():
        raise FileNotFoundError(f'Missing model: {MODEL}\nRun train_model.py first.')

    # Original LIAR TSV: column 1 = ID, column 2 = label, column 3 = statement.
    # Read only the two text fields required for this external evaluation.
    raw = pd.read_csv(
        DATA,
        sep='\t',
        header=None,
        usecols=[1, 2],
        dtype=str,
        quoting=csv.QUOTE_NONE,
        on_bad_lines='error',
        encoding='utf-8',
    )
    raw.columns = ['liar_label', 'statement']
    source_count = len(raw)

    raw['liar_label'] = raw['liar_label'].str.strip().str.lower()
    selected = raw[raw['liar_label'].isin(LABEL_MAP)].copy()
    selected = selected.dropna(subset=['statement'])
    selected['text_clean'] = selected['statement'].map(clean_text)
    selected = selected[selected['text_clean'].str.len() > 0].copy()

    if selected.empty or selected['liar_label'].nunique() < 2:
        raise ValueError('No usable binary LIAR evaluation examples with both classes.')

    labels = selected['liar_label'].map(LABEL_MAP).astype(int).to_numpy()
    model = joblib.load(MODEL)  # Only load the .pkl created locally by you.
    predictions = model.predict(selected['text_clean'].astype(str)).astype(int)

    report = classification_report(
        labels, predictions, labels=[0, 1],
        target_names=['FAKE', 'REAL'], output_dict=True, zero_division=0
    )
    counts = selected['liar_label'].value_counts().to_dict()
    confusion = confusion_matrix(labels, predictions, labels=[0, 1]).tolist()

    result = {
        'experiment': 'ISOT-trained NLP model tested without retraining on LIAR test.tsv',
        'domain': 'short political claims; different task from full-length news articles',
        'label_mapping': LABEL_MAP,
        'excluded_labels': ['barely-true', 'half-true', 'mostly-true'],
        'source_rows': int(source_count),
        'evaluated_rows': int(len(selected)),
        'excluded_or_invalid_rows': int(source_count - len(selected)),
        'original_label_counts_in_evaluation': counts,
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, average='macro', zero_division=0)),
        'majority_class_accuracy': float(max((labels == 0).mean(), (labels == 1).mean())),
        'classification_report': report,
        'confusion_matrix_rows_actual_fake_real': confusion,
    }

    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print('\nOUT-OF-DOMAIN LIAR TEST — NO RETRAINING')
    print('--------------------------------------')
    print('LIAR test rows:', source_count)
    print('Evaluated exact-label rows:', len(selected))
    print('Excluded/invalid rows:', source_count - len(selected))
    print('Label breakdown:', counts)
    print('Accuracy:         {:.4%}'.format(result['accuracy']))
    print('Balanced accuracy:{:.4%}'.format(result['balanced_accuracy']))
    print('Macro F1:         {:.4f}'.format(result['macro_f1']))
    print('Majority baseline:{:.4%}'.format(result['majority_class_accuracy']))
    print('\nCONFUSION MATRIX (rows actual FAKE/REAL; cols predicted FAKE/REAL)')
    print(confusion)
    print('\nCLASSIFICATION REPORT')
    print(classification_report(
        labels, predictions, labels=[0, 1],
        target_names=['FAKE', 'REAL'], zero_division=0
    ))
    print('Saved:', RESULTS.relative_to(BASE))
    print('CAUTION: Different domain and shorter text than ISOT; not a directly comparable score.')


if __name__ == '__main__':
    main()
