import pandas as pd

file_path = "data/raw/loan_approval_dataset.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Remove unnecessary spaces from column names
df.columns = df.columns.str.strip().str.lower()

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset dimensions (rows, columns) ---")
print(df.shape)

print("\n--- Column names ---")
print(df.columns.tolist())

print("\n--- Data types and non-missing counts ---")
df.info()

print("\n--- Missing values in each column ---")
print(df.isnull().sum())

print("\n--- Number of duplicate rows ---")
print(df.duplicated().sum())

print("\n--- Loan approval outcome counts ---")
print(
    df["loan_status"]
    .astype("string")
    .str.strip()
    .value_counts(dropna=False)
)

print("\n--- Numeric summary statistics ---")
print(df.describe())