import pandas as pd

df = pd.read_parquet("jobs_fall_2026.parquet")
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("Dates:", sorted(df["scraped_date"].unique().tolist()))
print("\nRows per date:")
print(df.groupby("scraped_date").size().to_string())