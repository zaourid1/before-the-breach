
from lightgbm import LGBMClassifier

from train import build_snapshot, FEATURES

# Same training and validation snapshots as before.
train = build_snapshot("2025-04")
validation = build_snapshot("2025-08")

X_train = train[FEATURES]
y_train = train["target"]

X_val = validation[FEATURES]
y_val = validation["target"]

# Gradient-boosted decision tree model.
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

# Predict positive-class ranking scores.
validation["lgbm_score"] = (
    model.predict_proba(X_val)[:, 1]
)

ranked = validation.sort_values(
    "lgbm_score",
    ascending=False
)

top500 = ranked.head(500)
hits = int(top500["target"].sum())

print("LIGHTGBM RESULTS")
print("Hits@500:", hits)
print("Precision@500:", hits / 500)

# Inspect feature importance.
importances = sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)

print("\nFEATURE IMPORTANCE")
for feature, importance in importances:
    print(f"{feature}: {importance}")
