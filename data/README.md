# Data needed to reproduce the experiments

No dataset is bundled in this repository. Create these folders locally:

- `data/raw/Fake.csv` and `data/raw/True.csv`: ISOT Fake and Real News Dataset (Kaggle: `clmentbisaillon/fake-and-real-news-dataset`).
- `data/external/liar/test.tsv`: original LIAR test split, Wang (2017), downloaded separately from `https://www.cs.ucsb.edu/~william/data/liar_dataset.zip`.

Then run `python prepare_dataset.py`, which writes the untracked `data/processed/news_dataset.csv`.

The `data/news_*.json` files are created during blockchain demo runs and contain submitted news text. Do not commit them without reviewing privacy and content rights.

Labels in this project: `0=FAKE`, `1=REAL`. LIAR evaluation excludes ambiguous labels and therefore uses only a subset of its test split.
