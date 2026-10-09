
import os
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

CHAIN_NAME = "fakenews"
STREAM_NAME = "news_records"

# MultiChain stores local RPC configuration here on Windows.
CHAIN_DIR = Path(os.environ["APPDATA"]) / "MultiChain" / CHAIN_NAME


def read_settings(path):
    settings = {}

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()

        if "=" in line:
            key, value = line.split("=", 1)
            settings[key.strip()] = value.strip()

    return settings


config = read_settings(CHAIN_DIR / "multichain.conf")
params = read_settings(CHAIN_DIR / "params.dat")

# RPC port is different from the blockchain's peer-to-peer port.
RPC_PORT = config.get("rpcport") or params["default-rpc-port"]
RPC_URL = f"http://127.0.0.1:{RPC_PORT}/"

AUTH = HTTPBasicAuth(
    config["rpcuser"],
    config["rpcpassword"]
)


def rpc_call(method, parameters=None):
    payload = {
        "jsonrpc": "1.0",
        "id": "python-test",
        "method": method,
        "params": parameters if parameters is not None else [],
        "chain_name": CHAIN_NAME
    }

    response = requests.post(
        RPC_URL,
        json=payload,
        auth=AUTH,
        timeout=15
    )

    response.raise_for_status()
    result = response.json()

    if result.get("error"):
        raise RuntimeError(result["error"])

    return result["result"]


if __name__ == "__main__":
    try:
        print("Connecting to MultiChain...")

        info = rpc_call("getinfo")

        print("CONNECTED SUCCESSFULLY")
        print("Blockchain:", info["chainname"])
        print("Current blocks:", info["blocks"])

        items = rpc_call(
            "liststreamkeyitems",
            [STREAM_NAME, "news001"]
        )

        if items:
            latest = items[-1]
            print("\nNEWS RECORD FOUND")
            print("Transaction ID:", latest["txid"])
            print("Keys:", latest["keys"])
            print("Stored data:", latest["data"])
            print("Confirmations:", latest["confirmations"])
        else:
            print("No news001 records found.")

    except Exception as error:
        print("Connection or RPC error:", error)
