
import argparse
import hashlib
import json
import re
import uuid

from pathlib import Path
from datetime import datetime, timezone

from multichain_client import rpc_call, STREAM_NAME
from predict_news import load_model
from prepare_dataset import clean_text


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def article_hash(headline, content):
    payload = json.dumps(
        {"headline": headline, "content": content},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return sha256_text(payload)


def submit_news():
    print("\nFAKE NEWS DETECTION SYSTEM")
    print("--------------------------")

    headline = input("Enter news headline: ").strip()
    content = input("Enter full news content: ").strip()

    if not headline or not content:
        print("ERROR: Headline and content are required.")
        return

    processed_text = clean_text(headline + " " + content)

    if len(processed_text.split()) < 25:
        print("ERROR: Enter at least 25 words of article text.")
        return

    print("\nLoading trained model...")
    model = load_model()

    # NLP prediction
    prediction = int(model.predict([processed_text])[0])

    probabilities = dict(zip(
        [int(c) for c in model.classes_],
        [float(p) for p in model.predict_proba(
            [processed_text]
        )[0]]
    ))

    prediction_label = "FAKE" if prediction == 0 else "REAL"
    model_score = probabilities[prediction]

    article_id = "news_" + uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    content_sha256 = sha256_text(content)
    full_sha256 = article_hash(headline, content)

    article = {
        "article_id": article_id,
        "headline": headline,
        "content": content,
        "created_at": timestamp
    }

    # Save the original article
    file_path = DATA_DIR / f"{article_id}.json"

    file_path.write_text(
        json.dumps(article, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    # Blockchain audit record
    record = {
        "article_id": article_id,
        "headline": headline,
        "prediction": prediction_label,
        "model_score": round(model_score, 6),
        "fake_score": round(probabilities[0], 6),
        "real_score": round(probabilities[1], 6),
        "model": "TFIDF_LogisticRegression_v1",
        "content_sha256": content_sha256,
        "article_sha256": full_sha256,
        "hash_scheme": "SHA256_canonical_JSON_v1",
        "submitted_at": timestamp,
        "status": "pending_review"
    }

    print("\nPublishing to MultiChain...")

    txid = rpc_call(
        "publish",
        [
            STREAM_NAME,
            article_id,
            {"json": record}
        ]
    )

    print("\nNEWS SUBMITTED SUCCESSFULLY")
    print("----------------------------")
    print("Article ID:", article_id)
    print("Prediction:", prediction_label)
    print(f"Model score: {model_score:.2%}")
    print("Review status: PENDING")
    print("Content SHA-256:", content_sha256)
    print("Article SHA-256:", full_sha256)
    print("Transaction ID:", txid)
    print("Local file:", file_path)
    print("\nThis is an ML prediction, not verified truth.")

    return article_id


def verify_article(article_id):
    if not re.fullmatch(r"news_[0-9a-f]{12}", article_id):
        print("ERROR: Invalid article ID.")
        return

    file_path = DATA_DIR / f"{article_id}.json"

    if not file_path.exists():
        print("ERROR: Local article not found.")
        return

    article = json.loads(
        file_path.read_text(encoding="utf-8")
    )

    if article.get("article_id") != article_id:
        print("ERROR: Local article ID mismatch.")
        return

    items = rpc_call(
        "liststreamkeyitems",
        [STREAM_NAME, article_id]
    )

    records = [
        item for item in items
        if item.get("data", {}).get("json", {}).get(
            "article_id"
        ) == article_id
        and "article_sha256" in item["data"]["json"]
    ]

    if len(records) != 1:
        print(
            "ERROR: Expected exactly one audit record.",
            "Found:", len(records)
        )
        return

    item = records[0]
    record = item["data"]["json"]

    content_matches = (
        sha256_text(article["content"])
        == record["content_sha256"]
    )

    article_matches = (
        article_hash(
            article["headline"],
            article["content"]
        ) == record["article_sha256"]
    )

    print("\nBLOCKCHAIN VERIFICATION RESULT")
    print("------------------------------")
    print("Article ID:", article_id)
    print("Prediction:", record["prediction"])
    print("Review status:", record["status"])
    print("Transaction ID:", item["txid"])
    print("Confirmations:", item["confirmations"])
    print("Content hash matches:", content_matches)
    print("Full article hash matches:", article_matches)

    if not content_matches or not article_matches:
        print("\nRESULT: TAMPERING DETECTED")
    elif item["confirmations"] < 1:
        print("\nRESULT: HASH MATCHED, AWAITING CONFIRMATION")
    else:
        print("\nRESULT: INTEGRITY VERIFIED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verify",
        metavar="ARTICLE_ID"
    )

    args = parser.parse_args()

    try:
        if args.verify:
            verify_article(args.verify)
        else:
            submit_news()
    except Exception as error:
        print("ERROR:", error)
