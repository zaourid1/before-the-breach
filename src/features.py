
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

signals = pd.read_csv(
    DATA / "exploit_signals_2025.csv"
)

kev = pd.read_csv(
    DATA / "known_exploited.csv",
    usecols=["cve_id", "date_added"]
)

print("Signals loaded:", len(signals))
print("KEV records loaded:", len(kev))

snapshot_month = "2025-04"

snapshot = signals.loc[
    signals["month"] == snapshot_month
].copy()

print("April observations:", len(snapshot))

kev["date_added"] = pd.to_datetime(
    kev["date_added"]
)

snapshot = snapshot.merge(
    kev,
    on="cve_id",
    how="left",
    validate="many_to_one"
)

prediction_date = pd.Timestamp("2025-04-30")
end_date = prediction_date + pd.Timedelta(days=90)

snapshot["target"] = (
    (snapshot["date_added"] > prediction_date)
    & (snapshot["date_added"] <= end_date)
).astype(int)

snapshot = snapshot.loc[
    snapshot["date_added"].isna()
    | (snapshot["date_added"] > prediction_date)
].copy()

print("\nTraining examples:", len(snapshot))

print("\nTarget distribution:")
print(snapshot["target"].value_counts())

print("\nPositive rate:")
print(snapshot["target"].mean())

feature_columns = [
    "epss",
    "exploitdb",
    "metasploit",
    "nuclei",
    "poc_github"
]

X = snapshot[feature_columns].fillna(0)
y = snapshot["target"]

print("\nFeature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

print("\nFirst five feature rows:")
print(X.head())


# Compare two ranking strategies on April 2025.

def evaluate_top500(data, name):
    top = data.head(500)

    hits = int(top["target"].sum())
    precision = hits / len(top)

    print(f"\n{name}")
    print(f"Hits in top 500: {hits}")
    print(f"Precision@500: {precision:.2%}")

    return hits


# Strategy 1: highest EPSS first.
epss_ranked = snapshot.sort_values(
    "epss",
    ascending=False
)

evaluate_top500(epss_ranked, "EPSS baseline")


# Strategy 2: public code first, then EPSS.
signal_columns = [
    "exploitdb",
    "metasploit",
    "nuclei",
    "poc_github"
]

snapshot["has_public_code"] = (
    snapshot[signal_columns]
    .fillna(0)
    .sum(axis=1) > 0
).astype(int)

code_ranked = snapshot.sort_values(
    ["has_public_code", "epss"],
    ascending=[False, False]
)

evaluate_top500(code_ranked, "Public code + EPSS baseline")
