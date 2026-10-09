
from pathlib import Path
import json

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent
DATASET = BASE_DIR / "data" / "processed" / "news_dataset.csv"
MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def train_model():
    print("\nLOADING DATASET...")
    data = pd.read_csv(DATASET)

    data = data.dropna(subset=["text_clean", "label"])

    if not set(data["label"].unique()) == {0, 1}:
        raise ValueError("Expected labels 0=Fake and 1=Real")

    if data["text_clean"].duplicated().any():
        raise ValueError("Duplicate articles found")

    print("Total records:", len(data))

    X = data["text_clean"].astype(str)
    y = data["label"].astype(int)

    # Stratified 70/30 split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    # TF-IDF is fitted only on training data.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            min_df=2,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42
        ))
    ])

    print("\nTRAINING MODEL...")
    model.fit(X_train, y_train)

    print("Training completed.")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        output_dict=True,
        zero_division=0
    )

    # Calculate ROC-AUC with Fake as positive class
    classes = list(
        model.named_steps["classifier"].classes_
    )
    fake_index = classes.index(0)

    fake_probabilities = model.predict_proba(
        X_test
    )[:, fake_index]

    auc = roc_auc_score(
        (y_test == 0).astype(int),
        fake_probabilities
    )

    print("\nMODEL EVALUATION RESULTS")
    print("-----------------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fake Precision: {report['Fake']['precision']:.4f}")
    print(f"Fake Recall: {report['Fake']['recall']:.4f}")
    print(f"Fake F1 Score: {report['Fake']['f1-score']:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Fake", "Real"],
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Fake", "Real"]
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Fake News Detection - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        RESULT_DIR / "confusion_matrix.png",
        dpi=200
    )
    plt.close()

    # Save trained pipeline
    joblib.dump(
        model,
        MODEL_DIR / "fake_news_model.pkl"
    )

    # Save experiment metrics
    metrics = {
        "dataset_records": len(data),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "accuracy": accuracy,
        "roc_auc": auc,
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    with open(
        RESULT_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    print("\nFILES SAVED SUCCESSFULLY")
    print("Model: models/fake_news_model.pkl")
    print("Metrics: results/metrics.json")
    print("Plot: results/confusion_matrix.png")


if __name__ == "__main__":
    train_model()
