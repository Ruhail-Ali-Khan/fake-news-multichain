
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from prepare_dataset import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_model.pkl"
DATASET_PATH = (
    BASE_DIR / "data" / "processed" / "news_dataset.csv"
)

LABELS = {
    0: "FAKE",
    1: "REAL"
}


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_news(model, text):
    prediction = int(model.predict([text])[0])

    probabilities = model.predict_proba([text])[0]

    class_probabilities = dict(
        zip(model.classes_, probabilities)
    )

    fake_probability = float(class_probabilities[0])
    real_probability = float(class_probabilities[1])

    print("\nNEWS CLASSIFICATION RESULT")
    print("--------------------------")
    print("Prediction:", LABELS[prediction])
    print(f"Fake probability: {fake_probability:.4f}")
    print(f"Real probability: {real_probability:.4f}")

    return prediction


def test_samples(model):
    data = pd.read_csv(DATASET_PATH)
    data = data.dropna(subset=["text_clean", "label"])

    X = data["text_clean"]
    y = data["label"]

    # Reconstruct the same held-out split used in training.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    for expected_label in [0, 1]:
        index = y_test[y_test == expected_label].index[0]
        article = X_test.loc[index]

        print("\n================================")
        print("ACTUAL LABEL:", LABELS[expected_label])
        print("ARTICLE PREVIEW:", article[:180], "...")

        predicted_label = predict_news(model, article)

        print(
            "CORRECT PREDICTION:",
            predicted_label == expected_label
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-samples",
        action="store_true"
    )

    parser.add_argument(
        "--title",
        default=""
    )

    parser.add_argument(
        "--text",
        default=""
    )

    args = parser.parse_args()

    model = load_model()

    if args.test_samples:
        test_samples(model)

    elif args.title or args.text:
        article_text = (
            clean_text(args.title)
            + " "
            + clean_text(args.text)
        ).strip()

        if len(article_text.split()) < 20:
            print(
                "WARNING: Very short text. "
                "Prediction may be unreliable."
            )

        predict_news(model, article_text)

    else:
        parser.error(
            "Provide --test-samples or article text."
        )
