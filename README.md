# BTR-Transformer

## Project Structure

```
tp1-llm/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── supermarket_products.csv
│   └── processed/
│       ├── dataset.parquet       # full cleaned/encoded dataset (for k-fold)
│       ├── train.parquet         # fixed split, for fast iteration
│       ├── valid.parquet         # fixed split
│       └── test.parquet          # final held-out set, touched once in both modes
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_results_analysis.ipynb   # plots for fixed split (architecture iteration)
│   └── 03_cv_analysis.ipynb        # plots for k-fold (final comparison / ablation)
├── src/
│   ├── data/
│   │   ├── preprocessing.py      # cleaning, one-hot encoding → produces dataset.parquet + test.parquet
│   │   ├── fixed_split.py        # generates train/valid.parquet from dataset.parquet
│   │   ├── splits.py             # GroupKFold(groups=query_id) over dataset.parquet
│   │   └── dataset.py            # PyTorch Dataset: groups by query_id, padding, mask
│   ├── models/
│   │   ├── text_encoder.py
│   │   ├── tabular_encoder.py
│   │   └── transformer.py
│   ├── train.py                  # --mode {fixed, cv} --fold N (only if mode=cv)
│   ├── evaluate.py               # PR-AUC, ROC-AUC; aggregates results if mode=cv
│   └── config.py                 # includes architecture variants to compare
├── experiments/
│   ├── dev/                      # fixed-split runs, during architecture iteration
│   │   ├── config_A/
│   │   ├── config_B/
│   │   └── ...
│   └── final_cv/                 # k-fold, only for the 2-3 finalist configs
│       ├── config_A/
│       │   ├── fold_0/ ... fold_4/
│       ├── config_B/
│       │   ├── fold_0/ ... fold_4/
│       └── results/
│           └── cv_summary.csv    # mean ± std per config, goes into the presentation
├── outputs/
│   └── figures/
├── slides/
│   └── presentacion.pptx
└── adr/                          # Architecture Decision Records
```
