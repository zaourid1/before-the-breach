from pathlib import Path
import pandas as pd

#Find the project root
ROOT = Path(__file__).resolve().parent.parent

#locate the benchmark csv
file_path = ROOT / "data" / "scoreboard_2026.csv"

#load the dataset into panada DF
df = pd.read_csv(file_path)

print("DATASET SHAPE: ")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())

print("\nFIRST FIVE ROWS:")
print(df.head())

print("\nMISSING VALUES:")
print(df.isna().sum())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nDUPLICATE CVE IDs:")
print(df["cve_id"].duplicated().sum())

#how are epss scores distributed across all vulnerabilities?
print("\nEPSS STATISTICS:")
print(df["epss"].describe())

print("\nTOP 10 HIGHEST EPSS SCORES:")

top10 = df.sort_values(
    by="epss",
    ascending=False
).head(10)

print(top10[["cve_id", "epss"]])