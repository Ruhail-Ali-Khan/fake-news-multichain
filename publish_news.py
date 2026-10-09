
import json
import hashlib
import uuid
from pathlib import Path
from datetime import datetime, timezone

from multichain_client import rpc_call, STREAM_NAME


def generate_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def publish_news(headline, content):
    article_id = "news_" + uuid.uuid4().hex[:12]

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    # Store the original article locally
    article = {
        "article_id": article_id,
        "headline": headline,
        "content": content,
        "created_at": timestamp
    }

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / f"{article_id}.json"

    file_path.write_text(
        json.dumps(article, indent=4),
        encoding="utf-8"
    )

    # Generate SHA-256 fingerprint
    content_hash = generate_hash(content)

    # Store metadata on the blockchain
    blockchain_record = {
        "article_id": article_id,
        "headline": headline,
        "content_sha256": content_hash,
        "submitted_at": timestamp,
        "status": "pending"
    }

    # Publish JSON through MultiChain RPC
    txid = rpc_call(
        "publish",
        [
            STREAM_NAME,
            article_id,
            {"json": blockchain_record}
        ]
    )

    print("\nNEWS PUBLISHED SUCCESSFULLY")
    print("---------------------------")
    print("Article ID:", article_id)
    print("Headline:", headline)
    print("SHA-256:", content_hash)
    print("Transaction ID:", txid)
    print("Local file:", file_path)

    return article_id, txid


if __name__ == "__main__":
    headline = "Sample News Article"

    content = (
        "This is a sample news article used "
        "to test the blockchain-based "
        "news verification system."
    )

    try:
        article_id, txid = publish_news(
            headline,
            content
        )

        print("\nChecking blockchain record...")

        items = rpc_call(
            "liststreamkeyitems",
            [STREAM_NAME, article_id]
        )

        if items:
            item = items[-1]
            stored = item["data"]["json"]

            print("Record retrieved successfully.")
            print("Status:", stored["status"])
            print(
                "Hash verified:",
                stored["content_sha256"]
                == generate_hash(content)
            )
            print(
                "Confirmations:",
                item["confirmations"]
            )
        else:
            print(
                "Transaction submitted. "
                "Wait for confirmation and query again."
            )

    except Exception as error:
        print("Error:", error)
