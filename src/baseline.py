from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

df = pd.read_csv(DATA / "scoreboard_2026.csv")

print(f"Loaded {len(df)} vulnerabilities")

#replacing missing EPSS values w/ 0 instead of predicting their scores
df["epss"] = df["epss"].fillna(0)

exploit_columns = [
    "exploitdb",
    "metasploit",
    "nuclei",
    "poc_github"
]

df["exploit_count"] = df[exploit_columns].sum(axis=1)

df["has_public_code"] = (df["exploit_count"] > 0).astype(int)

ranked = df.sort_values(
    by=["has_public_code", "epss"],
    ascending=[False, False]
)

top500 = ranked[["cve_id"]].head(500)

ranked[["cve_id"]]

output_path = ROOT / "top500.csv"

top500.to_csv(output_path, index=False)

print(f"Saved {len(top500)} vulnerabilities")
print(f"Output: {output_path}")

assert len(top500) == 500
assert top500["cve_id"].is_unique
assert top500.columns.tolist() == ["cve_id"]

print("All baseline checks passed!")

top_rows = ranked.head(500)

print("Vulnerabilities with public code:")
print(top_rows["has_public_code"].sum())

#which type of public exploit evidence occur most frquently in our top 500?
print("\nExploit signal counts in top 500:")

for column in exploit_columns:
    print(column, top_rows[column].sum())