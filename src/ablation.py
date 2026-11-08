
import pandas as pd
from lightgbm import LGBMClassifier

from train import build_snapshot

# Prepare the same datasets for every experiment.
months = ["2025-04", "2025-05", "2025-06"]

train = pd.concat(
    [build_snapshot(month) for month in months],
    ignore_index=True
)

validation = build_snapshot("2025-09")

base_features = [
    "epss",
    "exploitdb",
    "metasploit",
    "nuclei",
    "poc_github",
]

experiments = {
    "Original 5 features": base_features,
    "+ Vulnerability age": base_features + ["age_days"],
    "+ Exploit count": base_features + ["exploit_count"],
    "All 7 features": base_features + [
        "age_days",
        "exploit_count"
    ],
}

results = []

for name, features in experiments.items():

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
        train[features],
        train["target"]
    )

    scores = model.predict_proba(
        validation[features]
    )[:, 1]

    ranked = validation.assign(
        score=scores
    ).sort_values(
        "score",
        ascending=False,
        kind="stable"
    )

    hits = int(ranked.head(500)["target"].sum())

    results.append({
        "experiment": name,
        "features": len(features),
        "hits_at_500": hits,
        "precision_at_500": hits / 500
    })

results_df = pd.DataFrame(results)

print("\nABLATION STUDY RESULTS")
print(results_df.to_string(index=False))
