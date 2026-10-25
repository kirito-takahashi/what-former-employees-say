# What Former Employees Say

An IS507 Data, Statistical Models, and Information team project on Glassdoor reviews. We test whether former and current employees rate the same employer differently, whether their Cons text still differs at the same rating, and how many reviews an employer needs before that gap can be shown reliably. The decision behind it is whether an employer page should include a "What Former Employees Say" section, and for which companies. See [reports/midterm_report.md](reports/midterm_report.md) for details.

## Directory

```
what-former-employees-say/
├── README.md
├── requirements.txt
├── data/                                # not committed
│   ├── raw/
│   │   └── all_reviews 2.csv            # original Kaggle data
│   └── processed/
│       └── processed.csv                # cleaned data used by the notebooks
├── figures/
│   ├── cons_length.png
│   ├── rating.png
│   ├── reviews_per_employer.png
│   └── status.png
├── notebooks/
│   ├── 0.0-kt-data-prep.ipynb
│   └── 1.0-kt-sample-eda.ipynb
└── reports/
    └── midterm_report.md
```

## Setup

Requires Python 3.12 or later.

```bash
pip install -r requirements.txt
```
