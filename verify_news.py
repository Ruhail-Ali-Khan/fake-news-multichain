
import hashlib
import json
import sys
from pathlib import Path

from multichain_client import rpc_call, STREAM_NAME


DATA_DIR = Path(__file__).resolve().parent / "data"


def calculate_hash(content):
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def verify_news(article_id):
    file_path = DATA_DIR / f"{article_id}.json"

    if not file_path.exists():
        print("ERROR: Article file not found.")
        return False

    article = json.loads(
        file_path.read_text(encoding="utf-8")
    )

    if article.get("article_id") != article_id:
        print("ERROR: Article ID mismatch.")
        return False

    current_hash = calculate_hash(article["content"])

    # Retrieve the matching on-chain submission
    items = rpc_call(
        "liststreamkeyitems",
        [STREAM_NAME, article_id]
    )

    records = [
        item for item in items
        if item.get("data", {}).get("json", {}).get(
            "article_id"
        ) == article_id
        and "content_sha256" in item["data"]["json"]
    ]

    if len(records) != 1:
        print(
            "ERROR: Expected exactly one original "
            "hash record; found", len(records)
        )
        return False

    record = records[0]

    if record.get("confirmations", 0) < 1:
        print("WAIT: Blockchain record is not confirmed.")
        return False

    stored_hash = record["data"]["json"]["content_sha256"]

    print("\nBLOCKCHAIN INTEGRITY VERIFICATION")
    print("--------------------------------")
    print("Article ID:", article_id)
    print("Transaction:", record["txid"])
    print("Confirmations:", record["confirmations"])
    print("Original hash:", stored_hash)
    print("Current hash: ", current_hash)

    if current_hash == stored_hash:
        print("\nRESULT: INTEGRITY VERIFIED")
        print("Article content matches blockchain record.")
        return True

    print("\nRESULT: TAMPERING DETECTED")
    print("Article content differs from blockchain record.")
    return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_news.py <article_id>")
        sys.exit(2)

    try:
        verified = verify_news(sys.argv[1])
        sys.exit(0 if verified else 1)
    except Exception as error:
        print("Verification error:", error)
        sys.exit(1)
