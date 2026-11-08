from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

#load historical exploit indicators
signals = pd.read_csv(
    DATA / "exploit_signals_2025.csv"
)

#load confirmed exploitation records
kev = pd.read_csv(
    DATA / "known_exploited.csv"
)

print("HISTORICAL SIGNALS")
print("Shape:", signals.shape)
print("Columns:", signals.columns.tolist())
print(signals.head())

print("\nKNOWN EXPLOITED VULNERABILITIES")
print("Shape:", kev.shape)
print("Columns:", kev.columns.tolist())
print(kev.head())

print("\nKEV DATE RANGE")
print(kev["date_added"].min())
print(kev["date_added"].max())

print("\nUNIQUE CVEs IN SIGNALS:")
print(signals["cve_id"].nunique())

print("\nTOTAL MONTHLY RECORDS:")
print(len(signals))

print("\nMONTHS AVAILABLE:")
print(sorted(signals["month"].dropna().unique()))

print("\nKEV STATUS COUNTS:")
print(signals["in_kev"].value_counts())

print("\nKEV UNIQUE CVEs:")
print(kev["cve_id"].nunique())