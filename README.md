# What Former Employees Say

An IS507 Data, Statistical Models, and Information team project on Glassdoor reviews. We test whether former and current employees rate the same employer differently, whether their Cons text still differs at the same rating, and how many reviews an employer needs before that gap can be shown reliably. The decision behind it is whether an employer page should include a "What Former Employees Say" section, and for which companies. See [reports/midterm_report.md](reports/midterm_report.md) for details.

## Directory

```
what-former-employees-say/
├── README.md
├── requirements.txt
├── data/                                # not committed
│   ├── all_reviews.csv                  # original Kaggle data
│   ├── all_reviews.parquet              # parquet copy of the CSV
│   ├── all_reviews_clean.parquet        # written by EDA_BaselineModel
│   └── all_reviews_clean_test_firms.csv # written by EDA_BaselineModel
├── figures/
│   ├── cons_length.png
│   ├── rating.png
│   ├── reviews_per_employer.png
│   └── status.png
├── notebooks/
│   ├── Data_Exploration.ipynb
│   └── EDA_BaselineModel.ipynb
└── reports/
    └── midterm_report.md
```

## Setup

Requires Python 3.12 or later.

```bash
pip install -r requirements.txt
```
