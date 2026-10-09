# Static repository audit — 2026-10-09

Audited the 23 files supplied in `fake-news-source-audit.zip`: 12 Python scripts, 2 model artifacts, 7 results, and 2 dependency manifests. The additional README, ignore rules, tests, diagrams and checklist in this package were created during this audit; no existing application code was modified.

## Checks performed

- All 12 Python scripts parsed with Python `ast` (pass).
- All 5 JSON result documents parsed successfully and recorded numerical values are internally consistent with shared terminal outputs (pass). Two plots exist (pass).
- Confirmed no actual RPC password, private key, `.env`, or MultiChain node configuration in supplied ZIP (pass by limited keyword/pattern scan; **not** a complete secret-scanning guarantee).
- Trained model file ~1.44 MB and policy ~1.2 KB. Loading the pickle artifact was intentionally not performed during static audit.
- Prior user terminal evidence demonstrates successful publishing, confirming, retrieving, and hashing on their own Windows environment; this audit environment **did not** run a MultiChain node.

## Findings requiring accurate disclosure

- **High | Methodological scope:** The paper discusses Hyperledger Fabric/Composer and deep RL; this implementation uses MultiChain plus a one-step, 20-bin tabular Q-learning/contextual-bandit policy.
- **High | Claimed authoritativeness:** No actual reviewer/validator workflow, verified identity, multi-party approval, smart contract, or consensus comparison is present. The `pending_review` field is never adjudicated.
- **High | Generalization:** ISOT accuracy 98.50%; LIAR accuracy 52.80% across 553 label-filtered statements (majority baseline 61.84%). Avoid general claims about real-world misinformation accuracy.
- **Medium | Baseline comparison:** Training-calibrated confidence review produced 16 mistakes and 1,071 referrals; tabular RL produced 21 mistakes and 1,055 referrals. Review rates differ slightly; RL has no demonstrated advantage.
- **Medium | Preprocessing audit:** ISOT preprocessing removed 5,957 articles from original 44,898; class-skewed deletions were not separately audited by cause. Source and writing-style leakage remain possibilities.
- **Medium | Integrity semantics:** The integrated system hashes only the headline/content (and separately content), not the complete audit metadata, model artifact, or local JSON envelope. A confirmed hash match is **not** independent source authentication.
- **Medium | Reproducibility:** ZIP excluded raw and processed datasets, as it should. Readme/data instructions have been added, but a third-party fresh-clone end-to-end run is still outstanding.
- **Medium | Security:** The RPC client reads local node credentials and uses HTTP to loopback, which is appropriate for the tested local prototype but not ready for remote deployment. Additional wallets/roles are not implemented.
- **Low | Operational:** The basic `multichain_client.py` demo assumes a key `news001` exists; a brand-new chain without this demo item prints a no-record message.
- **Low | Artifacts:** User-supplied screenshots were not in the source ZIP; the evidence checklist lists which genuine screenshots to add to an academic submission.

## Remaining release gates

1. On the original Windows machine: run the unit tests and fresh-clone setup; verify `multichain-cli fakenews getinfo`, read/subscribe/publish/verify on a copy or testing chain.
2. Validate `git status`, `git check-ignore` and scan tracked files for secrets before any public push.
3. Capture and caption screenshots with local private information reviewed.
4. Submit the original paper separately as required by the instructor; README links to its DOI, the original PDF is not bundled in this code package.
5. Write the assigned academic report independently from genuine implementation evidence; verify similarity and institution AI-use policy. AI-authorship percentages cannot be guaranteed by this tool.
