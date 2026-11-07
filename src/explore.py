from pathlib import Path
import pandas as pd


from pathlib import Path
import pandas as pd

# Locate the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Construct the path to our dataset.
data_path = PROJECT_ROOT / "data" / "scoreboard_2026.csv"

# Read the CSV into a pandas DataFrame.
df = pd.read_csv(data_path)

# Explore the data.
print("Number of rows and columns:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five vulnerabilities:")
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())
