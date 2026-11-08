
import pandas as pd
from lightgbm import LGBMClassifier

from train import build_snapshot, FEATURES

# 1. Build three historical training snapshots.
months = ["2025-04", "2025-05", "2025-06"]

training_frames = []

for month in months:
    frame = build_snapshot(month)
    training_frames.append(frame)

    print(
        month,
        "rows:", len(frame),
        "positives:", frame["target"].sum()
    )

# 2. Combine the snapshots vertically.
train = pd.concat(
    training_frames,
    ignore_index=True
)

# 3. Build a later validation snapshot.
validation = build_snapshot("2025-09")

print("\nTotal training rows:", len(train))
print("Total training positives:", train["target"].sum())
print("Validation positives:", validation["target"].sum())

# 4. Select features and labels.
X_train = train[FEATURES]
y_train = train["target"]

X_val = validation[FEATURES]

# 5. Train LightGBM.
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

model.fit(X_train, y_train)

# 6. Evaluate the learned ranking.
validation["ml_score"] = model.predict_proba(
    X_val
)[:, 1]

ranked = validation.sort_values(
    "ml_score",
    ascending=False,
    kind="stable"
)

hits = int(ranked.head(500)["target"].sum())

print("\nMULTI-MONTH LIGHTGBM")
print("Hits@500:", hits)
print("Precision@500:", hits / 500)

# 7. Compare against EPSS on the SAME validation set.
epss_ranked = validation.sort_values(
    "epss",
    ascending=False,
    kind="stable"
)

epss_hits = int(
    epss_ranked.head(500)["target"].sum()
)

print("EPSS baseline Hits@500:", epss_hits)

positive_rows = train[train["target"] == 1]

print("\nPOSITIVE LABEL ANALYSIS")
print("Positive observations:", len(positive_rows))
print("Unique positive CVEs:", positive_rows["cve_id"].nunique())
