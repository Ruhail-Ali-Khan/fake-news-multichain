# Original screenshot evidence checklist

Use screenshots from your own command prompt. Do not fabricate terminal output. Use short captions and explain the visible result under each figure. Do not expose RPC passwords.

| Proposed figure | Real command/output to capture | Observation to write in your own words |
|---|---|---|
| 1 | `multichain-cli.exe fakenews getinfo` | Chain exists, current block height, no RPC errors |
| 2 | `multichain-cli.exe fakenews liststreams` | `news_records`, write restriction, subscription |
| 3 | `multichain-cli.exe fakenews liststreamitems news_records` | Article item and confirmed txid |
| 4 | `python multichain_client.py` | Python JSON-RPC successfully reads the stream |
| 5 | `python publish_news.py` | News SHA-256 and txid published |
| 6 | `python verify_news.py <id>` after editing article | Local content hash differs; tampering flagged |
| 7 | `python verify_news.py <id>` after restoration | Content hash matches confirmed record |
| 8 | `python prepare_dataset.py` | Record counts after preprocessing |
| 9 | `python train_model.py` + `results/confusion_matrix.png` | Held-out ISOT baseline performance |
| 10 | `python predict_news.py --test-samples` | Two held-out inference examples |
| 11 | `python news_pipeline.py --verify <id>` | On-chain full-article hash verification |
| 12 | `python train_rl.py` + `results/rl_policy_actions.png` | Q-learning policy + validation metrics |
| 13 | `python rl_news_pipeline.py` | NLP prediction, Q-learning action, published txid |
| 14 | `python compare_triage.py` | RL versus budget-matched confidence triage |
| 15 | `python calibrated_triage.py` | Training-only calibrated threshold results |
| 16 | `python evaluate_liar_external.py` | Out-of-domain LIAR performance |

Screenshots are not included in this audited-source archive. Review local paths and identifying information before adding them to any **public** GitHub repository.
