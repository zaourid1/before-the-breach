
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CONFIGURATION
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Only actual public exploit indicators.
EXPLOIT_COLUMNS = [
    "exploitdb",
    "metasploit",
    "nuclei",
    "poc_github",
]

# All features used to train the model.
FEATURES = [
    "epss",
    "exploitdb",
    "metasploit",
    "nuclei",
    "poc_github",
    "age_days",       # FIX: Added missing comma
    "exploit_count",
]


# ==========================================================
# 2. LOAD DATA
# ==========================================================

signals = pd.read_csv(
    DATA / "exploit_signals_2025.csv"
)

kev = pd.read_csv(
    DATA / "known_exploited.csv",
    usecols=["cve_id", "date_added"]
)

publications = pd.read_csv(
    DATA / "vulnerabilities.csv",
    usecols=["cve_id", "published"]
)

# Convert strings into actual datetime objects.
kev["date_added"] = pd.to_datetime(
    kev["date_added"],
    errors="coerce"
)

publications["published"] = pd.to_datetime(
    publications["published"],
    errors="coerce"
)

# Each CVE should have only one matching record.
assert kev["cve_id"].is_unique
assert publications["cve_id"].is_unique


# ==========================================================
# 3. BUILD A HISTORICAL SNAPSHOT
# ==========================================================

def build_snapshot(month):

    # Select records available in the requested month.
    snapshot = signals.loc[
        signals["month"] == month
    ].copy()

    # Attach KEV addition dates.
    snapshot = snapshot.merge(
        kev,
        on="cve_id",
        how="left",
        validate="many_to_one"
    )

    # Attach publication dates.
    snapshot = snapshot.merge(
        publications,
        on="cve_id",
        how="left",
        validate="many_to_one"
    )

    # Last day of the selected month.
    prediction_date = (
        pd.Period(month)
        .to_timestamp(how="end")
        .normalize()
    )

    # Predict KEV additions within the next 90 days.
    end_date = prediction_date + pd.Timedelta(days=90)

    # We only have KEV records through December 2025.
    assert end_date <= pd.Timestamp("2025-12-29"), (
        "Prediction window exceeds available KEV labels."
    )

    # ------------------------------------------------------
    # FEATURE ENGINEERING: VULNERABILITY AGE
    # ------------------------------------------------------

    snapshot["age_days"] = (
        prediction_date - snapshot["published"]
    ).dt.days

    # Remove CVEs with missing or future publication dates.
    snapshot = snapshot.loc[
        snapshot["age_days"].notna()
        & (snapshot["age_days"] >= 0)
    ].copy()

    # ------------------------------------------------------
    # REMOVE ALREADY EXPLOITED CVEs
    # ------------------------------------------------------

    # A CVE already on KEV cannot become a new KEV addition.
    snapshot = snapshot.loc[
        snapshot["date_added"].isna()
        | (snapshot["date_added"] > prediction_date)
    ].copy()

    # ------------------------------------------------------
    # CREATE TARGET LABEL
    # ------------------------------------------------------

    snapshot["target"] = (
        (snapshot["date_added"] > prediction_date)
        & (snapshot["date_added"] <= end_date)
    ).astype(int)

    # ------------------------------------------------------
    # FEATURE ENGINEERING: PUBLIC EXPLOIT CODE
    # ------------------------------------------------------

    snapshot["exploit_count"] = (
        snapshot[EXPLOIT_COLUMNS]
        .fillna(0)
        .sum(axis=1)
    )

    snapshot["has_public_code"] = (
        snapshot["exploit_count"] > 0
    ).astype(int)

    # ------------------------------------------------------
    # HANDLE MISSING FEATURES
    # ------------------------------------------------------

    snapshot[FEATURES] = (
        snapshot[FEATURES].fillna(0)
    )

    # Verify that every required feature exists.
    assert all(
        column in snapshot.columns
        for column in FEATURES
    )

    assert not snapshot[FEATURES].isna().any().any()
    assert snapshot["cve_id"].is_unique

    return snapshot


# ==========================================================
# 4. EVALUATION FUNCTION
# ==========================================================

def evaluate(data, score_column, name):

    ranked = data.sort_values(
        score_column,
        ascending=False,
        kind="stable"
    )

    top500 = ranked.head(500)

    hits = int(top500["target"].sum())

    precision = hits / len(top500)

    print(f"\n{name}")
    print(f"Hits@500: {hits}")
    print(f"Precision@500: {precision:.4f}")

    return hits


# ==========================================================
# 5. MAIN TRAINING EXPERIMENT
# ==========================================================

def main():

    # Create historical training and validation datasets.
    train = build_snapshot("2025-04")
    validation = build_snapshot("2025-08")

    print("TRAINING")
    print("Rows:", len(train))
    print("Positives:", train["target"].sum())

    print("\nVALIDATION")
    print("Rows:", len(validation))
    print("Positives:", validation["target"].sum())

    # Separate features (X) from labels (y).
    X_train = train[FEATURES]
    y_train = train["target"]

    X_val = validation[FEATURES]

    # ------------------------------------------------------
    # TRAIN LOGISTIC REGRESSION
    # ------------------------------------------------------

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    print("\nModel trained successfully!")

    # Get scores for the positive class.
    validation["ml_score"] = (
        model.predict_proba(X_val)[:, 1]
    )

    # ------------------------------------------------------
    # COMPARE AGAINST BASELINES
    # ------------------------------------------------------

    # Public-code indicator is already created by
    # build_snapshot() using the correct four columns.
    baseline_ranked = validation.sort_values(
        ["has_public_code", "epss"],
        ascending=[False, False],
        kind="stable"
    )

    print("\nVALIDATION RESULTS")

    evaluate(
        validation,
        "epss",
        "EPSS baseline"
    )

    baseline_hits = int(
        baseline_ranked.head(500)["target"].sum()
    )

    print(
        "\nPublic code + EPSS baseline:",
        baseline_hits,
        "hits in top 500"
    )

    evaluate(
        validation,
        "ml_score",
        "Logistic regression"
    )

    # ------------------------------------------------------
    # INSPECT LEARNED COEFFICIENTS
    # ------------------------------------------------------

    classifier = model.named_steps["logisticregression"]

    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": classifier.coef_[0]
    })

    coefficients = coefficients.sort_values(
        "coefficient",
        ascending=False
    )

    print("\nLEARNED FEATURE COEFFICIENTS:")
    print(coefficients.to_string(index=False))

    print("\nINTERCEPT:")
    print(classifier.intercept_[0])

    # ------------------------------------------------------
    # COMPARE TOP-500 SELECTIONS
    # ------------------------------------------------------

    epss_top = set(
        validation.nlargest(500, "epss")["cve_id"]
    )

    ml_top = set(
        validation.nlargest(500, "ml_score")["cve_id"]
    )

    overlap = epss_top & ml_top

    print("\nTOP-500 OVERLAP")
    print("Shared CVEs:", len(overlap))
    print("EPSS-only CVEs:", len(epss_top - ml_top))
    print("ML-only CVEs:", len(ml_top - epss_top))


# ==========================================================
# 6. ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
