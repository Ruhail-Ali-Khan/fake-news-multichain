# Fake News Detection: NLP + Q-learning + MultiChain

**Scope statement:** This is a Windows proof of concept, **not** an exact reproduction of the original research paper. The paper discusses Hyperledger Fabric/Composer and deep reinforcement learning; this project instead uses MultiChain 2.3.3 JSON-RPC and a one-step tabular Q-learning review policy. No multi-organization validator approval workflow, deployed smart contract, or independent fact verification is implemented. “FAKE” and “REAL” are model-generated labels, not established truth.

## Features

- Text preprocessing, TF-IDF and Logistic Regression binary news classification.
- One-step Q-learning policy that recommends `FAKE`, `REAL`, or simulated `REVIEW` based on model score.
- MultiChain permissioned stream (`news_records`) for recording predictions, timestamps, and SHA-256 hashes.
- Local article storage and hash-based integrity verification after blockchain confirmation.
- Controlled NLP/RL comparison, confidence-based triage calibration, external LIAR evaluation.

## System architecture and flowcharts

[Source architecture diagram](docs/diagrams/system_architecture.mmd) and [MultiChain implementation workflow](docs/diagrams/multichain_workflow.mmd) use Mermaid syntax. The main workflow is:

```mermaid
flowchart LR
A[News Article] --> B[NLP Model]
B --> C[Q-learning Review Policy]
C --> D[SHA-256 Audit Metadata]
D --> E[MultiChain news_records]
E --> F[Retrieve and Check Hash]
```

## Requirements

- Windows with MultiChain Community 2.3.3 installed and `multichaind.exe`, `multichain-cli.exe`, and `multichain-util.exe` available.
- Python **3.11** (the environment used for the reported experiments).
- Permissions to run a local MultiChain node; a working `fakenews` chain with a `news_records` stream.
- ISOT dataset downloaded separately if retraining; original LIAR `test.tsv` for the external experiment.

## Setup on a fresh Windows machine

In a Command Prompt in the project folder:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python -m pip check
```

`requirements.txt` lists the direct dependencies. `requirements-lock.txt` captures the original environment's full `pip freeze`; it may need regeneration for a different Python version or OS. Do **not** execute untrusted `.pkl` files; `joblib.load` deserializes Python objects. The included model file is part of the research snapshot and can alternatively be regenerated locally from the dataset.

### Create and start MultiChain

From the directory containing the MultiChain executables:

```cmd
multichain-util.exe create fakenews
multichaind.exe fakenews -daemon
multichain-cli.exe fakenews getinfo
multichain-cli.exe fakenews liststreams
```

`create` is a one-time command. **Do not recreate an existing chain.** If `news_records` does not yet exist, create it and subscribe:

```cmd
multichain-cli.exe fakenews create stream news_records "{\"restrict\":\"write\"}"
multichain-cli.exe fakenews subscribe news_records
multichain-cli.exe fakenews liststreams
```

The local node must have the appropriate rights to publish to the write-restricted stream. Check account/network permissions if publishing fails. This demo demonstrates a local permissioned stream, **not** a tested multi-validator governance system.

`multichain_client.py` reads local RPC credentials from `%APPDATA%\MultiChain\fakenews\multichain.conf` and the RPC port from chain config/params. **Never commit** configuration files or credentials. The code connects only to `127.0.0.1`.

### Get the training and evaluation datasets

Read [`data/README.md`](data/README.md), download `Fake.csv` and `True.csv` into `data/raw/`, then:

```cmd
python prepare_dataset.py
python train_model.py
python train_rl.py
```

Note: dataset files are intentionally excluded from GitHub. The saved `.pkl` and `.npz` are included for a quick demonstration, but rerunning training is essential for a fresh reproducibility check.

### Run the demonstration

```cmd
python multichain_client.py
python predict_news.py --test-samples
python news_pipeline.py
python rl_news_pipeline.py
```

The last two commands prompt for a headline and an article (minimum 25 words). They save `data/news_<id>.json` and publish an audit entry to the `news_records` MultiChain stream.

Copy the printed article ID, then verify after confirmation:

```cmd
python news_pipeline.py --verify news_<your12hexcharacters>
```

For the earlier hash-only demonstration, run `python publish_news.py`, followed by `python verify_news.py news_<id>`; it verifies the content hash only. The integrated verification command checks both headline/content and content-only hashes. A matching hash proves agreement with the blockchain snapshot, **not news truthfulness**.

### Reproduce the comparisons and robustness test

```cmd
python compare_triage.py
python calibrated_triage.py
python evaluate_liar_external.py
python -m unittest discover -s tests -v
```

The external evaluation requires `data/external/liar/test.tsv`. `compare_triage.py` retrospectively matches review budgets on test scores; `calibrated_triage.py` freezes its threshold using training-partition calibration. Neither simulated review case has been reviewed by a human.


## Code map

| Script | Role |
|---|---|
| `multichain_client.py` | Local JSON-RPC communication |
| `publish_news.py`, `verify_news.py` | Basic publish/hash experiment |
| `prepare_dataset.py` | ISOT cleaning and deduplication |
| `train_model.py`, `predict_news.py` | NLP training and inference |
| `news_pipeline.py` | NLP prediction, publication and verification |
| `train_rl.py` | Tabular Q-learning policy and evaluation |
| `rl_news_pipeline.py` | NLP + RL + MultiChain demonstration |
| `compare_triage.py`, `calibrated_triage.py` | Baseline comparisons |
| `evaluate_liar_external.py` | Out-of-domain evaluation |

## Security, reproducibility and research limitations

1. RPC credentials stay on the local machine; never commit `%APPDATA%\MultiChain\...` files, `.env`, or private keys.
2. The stream has `restrict.write=true`, but this demo does not implement separate publisher/validator wallets, identity verification, approval voting, or smart contracts.
3. A `pending_review` status is written, but no actual reviewer interface or validated final decision exists.
4. Local JSON files are **not** distributed or durable off-chain storage. Deleting a local article prevents this demo's content-hash check from reconstructing the original text.
5. Blockchain hashes detect differences between a local copy and the stored digest; they do not validate the original source, protect local files from deletion, or prove the truth of the text.
6. Source/style cues and content duplicates may inflate performance on ISOT. Only the specific included preprocessing and held-out evaluation have been tested.
7. The Q-learning policy is one-step tabular RL, **not** a deep RL reproduction.
8. No independent multi-node, access-control, throughput, or latency benchmark was run. Do not reuse numerical performance claims from the original paper as measurements from this code.

See [`docs/SCREENSHOT_CHECKLIST.md`](docs/SCREENSHOT_CHECKLIST.md) for genuine execution evidence needed in the implementation report.
