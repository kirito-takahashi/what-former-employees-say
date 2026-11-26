# What Former Employees Say

An IS507 Data, Statistical Models, and Information team project on Glassdoor reviews. We test whether former and current employees rate the same employer differently, whether their Cons text still differs at the same rating, and how many reviews an employer needs before that gap can be shown reliably. The decision behind it is whether an employer page should include a "What Former Employees Say" section, and for which companies.

The full analysis is in [final/final_analysis.ipynb](final/final_analysis.ipynb). See [reports/midterm_report.md](reports/midterm_report.md) for the study design.

## Directory

```
what-former-employees-say/
├── README.md
├── requirements.txt
├── data/
│   └── final-analysis/
│       ├── raw/
│       │   └── all_reviews 2.csv
│       └── processed/
│           ├── processed.parquet
│           └── rq2/
├── final/
│   ├── final_analysis.ipynb
│   └── final_report.md
├── experiment/
├── figures/
│   ├── final_report/
│   └── midterm_report/
└── reports/
    ├── related_work.md
    └── midterm_report.md
```

- `data/` is not committed. `all_reviews 2.csv` is the original Kaggle data, and `processed/` is written by the final notebook (`rq2/` holds the sentence embeddings for the appendix).
- `final/final_analysis.ipynb` covers data prep, EDA, RQ1-3, the conclusion and the appendix.
- `figures/` holds the figures embedded in each report, one folder per report.
- `experiment/` holds earlier exploratory notebooks and scripts (see [Experiment Files](#experiment-files)).

## How to Run

The final notebook runs on Google Colab (Python 3.12 or later).

1. Copy this repository to `MyDrive/what-former-employees-say/` in Google Drive.
2. Put `all_reviews 2.csv` in `data/final-analysis/raw/`.
3. Add a Hugging Face token as `HF_TOKEN` in Colab Secrets. The appendix uses it to download EmbeddingGemma.
4. Select a GPU runtime. Only the appendix needs it.
5. Open `final/final_analysis.ipynb` and run all cells. The first cell mounts Drive and installs `requirements.txt`.

The notebook writes `processed.parquet` and the embedding files to `data/final-analysis/processed/`. If the embedding files already exist, it loads them instead of encoding again.

## Experiment Files

Files in `experiment/` are named `<step>-<author>-<topic>`. The step number shows the part of the analysis: 0 data prep, 1 EDA, 2 RQ1, 3 RQ2 and 4 RQ3. The author initials are:

| Initials | Author |
| :-- | :-- |
| kt | `kirito-takahashi` |
| yz | `glow31011` |

These files read the older `data/processed/` and are kept for reference only.
