
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from train import build_snapshot, FEATURES, EXPLOIT_COLUMNS


# ======================================================
# 1. PROJECT PATHS
# ======================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ======================================================
# 2. TRAIN THE FINAL MODEL
# ======================================================

# Include all these historical months for the final fit.
# Their 90-day label windows end by December 29, 2025.
training_months = [
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
]

training_frames = []

for month in training_months:
    frame = build_snapshot(month)
    training_frames.append(frame)

train = pd.concat(
    training_frames,
    ignore_index=True
)

print("Training rows:", len(train))
print("Positive observations:", train["target"].sum())
print(
    "Unique positive CVEs:",
    train.loc[
        train["target"] == 1, "cve_id"
    ].nunique()
)

# Use the seven-feature model selected during
# our historical ablation experiment.
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

print("Final model trained successfully!")


# ======================================================
# 3. LOAD THE 2026 BENCHMARK
# ======================================================

scoreboard = pd.read_csv(
    DATA / "scoreboard_2026.csv"
)

assert len(scoreboard) == 72670
assert scoreboard["cve_id"].is_unique

scoreboard["published"] = pd.to_datetime(
    scoreboard["published"],
    errors="coerce"
)

# Snapshot date = last calendar day of its month.
snapshot_dates = pd.PeriodIndex(
    scoreboard["snapshot_month"],
    freq="M"
).to_timestamp(how="end").normalize()

# Age at the historical prediction snapshot.
scoreboard["age_days"] = (
    snapshot_dates - scoreboard["published"]
).dt.days


# ======================================================
# 4. CREATE THE SAME SEVEN FEATURES
# ======================================================

for column in ["epss"] + EXPLOIT_COLUMNS:
    scoreboard[column] = pd.to_numeric(
        scoreboard[column],
        errors="coerce"
    ).fillna(0)

scoreboard["exploit_count"] = (
    scoreboard[EXPLOIT_COLUMNS].sum(axis=1)
)

# Verify publication dates are usable.
assert scoreboard["age_days"].notna().all()
assert (scoreboard["age_days"] >= 0).all()

# Verify all seven feature columns are available.
assert not scoreboard[FEATURES].isna().any().any()

print("\nInference features:")
print(scoreboard[FEATURES].head())


# ======================================================
# 5. GENERATE ML SCORES
# ======================================================

scoreboard["ml_score"] = model.predict_proba(
    scoreboard[FEATURES]
)[:, 1]

assert scoreboard["ml_score"].notna().all()


# ======================================================
# 6. RANK VULNERABILITIES
# ======================================================

ranked = scoreboard.sort_values(
    ["ml_score", "epss", "cve_id"],
    ascending=[False, False, True],
    kind="stable"
)

top500 = ranked[["cve_id"]].head(500)


# ======================================================
# 7. EXPORT SUBMISSION
# ======================================================

top500.to_csv(
    ROOT / "top500.csv",
    index=False
)

# Save ranking details for our future dashboard.
ranked.to_csv(
    ROOT / "ranking_details_2026.csv",
    index=False
)

assert len(top500) == 500
assert top500["cve_id"].is_unique
assert top500.columns.tolist() == ["cve_id"]

print("\nFinal top500.csv created!")
print("Top five predictions:")
print(ranked[["cve_id", "ml_score"]].head())

print("\nAll submission checks passed!")
