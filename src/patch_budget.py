
"""Evaluate vulnerability rankings at different investigation budgets."""

from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from train import build_snapshot, FEATURES


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "results"
OUTPUT.mkdir(exist_ok=True)

TRAINING_MONTHS = [
    "2025-04",
    "2025-05",
    "2025-06",
]

BUDGETS = [25, 50, 100, 250, 500, 750, 1000]


# --------------------------------------------------
# 1. PREPARE TRAINING AND VALIDATION DATA
# --------------------------------------------------

train = pd.concat(
    [build_snapshot(month) for month in TRAINING_MONTHS],
    ignore_index=True
)

validation = build_snapshot("2025-09")

print("Training observations:", len(train))
print("Positive training observations:", train["target"].sum())
print(
    "Unique positive training CVEs:",
    train.loc[train["target"] == 1, "cve_id"].nunique()
)

print("\nValidation observations:", len(validation))
print("Validation positives:", validation["target"].sum())


# --------------------------------------------------
# 2. TRAIN LIGHTGBM
# --------------------------------------------------

model = LGBMClassifier(
    objective="binary",
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=7,
    max_depth=4,
    min_child_samples=100,
    reg_lambda=5.0,
    random_state=42,
    verbosity=-1
)

model.fit(
    train[FEATURES],
    train["target"]
)

validation["ml_score"] = model.predict_proba(
    validation[FEATURES]
)[:, 1]


# --------------------------------------------------
# AUDIT: RECENTLY PUBLISHED VULNERABILITIES
# --------------------------------------------------

# Keep CVEs published within 90 days of the snapshot.
recent = validation.loc[
    validation["age_days"] <= 90
].copy()

# Rank CVEs by LightGBM and EPSS.
# Use CVE ID as a consistent tie-breaker.
recent = recent.sort_values(
    ["ml_score", "cve_id"],
    ascending=[False, True]
)

recent["ml_rank"] = range(1, len(recent) + 1)

epss_order = recent.sort_values(
    ["epss", "cve_id"],
    ascending=[False, True]
).copy()

epss_order["epss_rank"] = range(
    1, len(epss_order) + 1
)

recent = recent.merge(
    epss_order[["cve_id", "epss_rank"]],
    on="cve_id",
    how="left",
    validate="one_to_one"
)

# Select only CVEs subsequently added to KEV.
positives = recent.loc[
    recent["target"] == 1
].copy()

columns = [
    "cve_id",
    "age_days",
    "date_added",
    "epss",
    "ml_rank",
    "epss_rank",
    "ml_score"
]

positives = positives.sort_values("ml_rank")

print("\n90-DAY COHORT — POSITIVE CVE AUDIT")
print(positives[columns].to_string(index=False))

positives[columns].to_csv(
    OUTPUT / "recent_positive_audit.csv",
    index=False
)

print("\nTotal positives:", len(positives))
print(
    "LightGBM Hits@500:",
    (positives["ml_rank"] <= 500).sum()
)
print(
    "EPSS Hits@500:",
    (positives["epss_rank"] <= 500).sum()
)

# --------------------------------------------------
# 3. EVALUATE A RANKING AT DIFFERENT BUDGETS
# --------------------------------------------------

def evaluate_budget(data, score_column, budget):
    """Return future KEV additions among the top K CVEs."""

    ranked = data.sort_values(
        [score_column, "cve_id"],
        ascending=[False, True],
        kind="stable"
    )

    top = ranked.head(budget)

    return int(top["target"].sum())


# --------------------------------------------------
# 4. DEFINE OUR EVALUATION POPULATIONS
# --------------------------------------------------

cohorts = {
    "All eligible CVEs": validation,
    "Published within 180 days": validation.loc[
        validation["age_days"] <= 180
    ].copy(),
    "Published within 90 days": validation.loc[
        validation["age_days"] <= 90
    ].copy(),
    "Published within 30 days": validation.loc[
        validation["age_days"] <= 30
    ].copy(),
}


# --------------------------------------------------
# 5. COMPUTE REAL PATCH-BUDGET CURVES
# --------------------------------------------------

results = []
cohort_summary = []

for cohort_name, cohort in cohorts.items():

    candidates = len(cohort)
    positives = int(cohort["target"].sum())

    cohort_summary.append({
        "cohort": cohort_name,
        "candidates": candidates,
        "positive_cves": positives,
        "positive_rate": (
            positives / candidates if candidates else 0
        ),
    })

    print(f"\n{cohort_name}")
    print("Candidates:", candidates)
    print("Future KEV additions:", positives)

    if candidates == 0:
        continue

    for budget in BUDGETS:
        if budget > candidates:
            continue

        epss_hits = evaluate_budget(
            cohort, "epss", budget
        )

        ml_hits = evaluate_budget(
            cohort, "ml_score", budget
        )

        results.append({
            "cohort": cohort_name,
            "budget": budget,
            "candidates": candidates,
            "positive_cves": positives,
            "epss_hits": epss_hits,
            "lightgbm_hits": ml_hits,
            "difference": ml_hits - epss_hits,
            "epss_recall": (
                epss_hits / positives if positives else 0
            ),
            "lightgbm_recall": (
                ml_hits / positives if positives else 0
            ),
        })

        print(
            f"Top {budget}: "
            f"EPSS={epss_hits}, "
            f"LightGBM={ml_hits}, "
            f"Difference={ml_hits-epss_hits:+d}"
        )


# --------------------------------------------------
# 6. SAVE REPRODUCIBLE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)
summary_df = pd.DataFrame(cohort_summary)

results_df.to_csv(
    OUTPUT / "patch_budget_results.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT / "cohort_summary.csv",
    index=False
)

print("\nFiles saved:")
print(OUTPUT / "patch_budget_results.csv")
print(OUTPUT / "cohort_summary.csv")
