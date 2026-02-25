import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Print first 5 rows
print("First 5 rows:")
print(df.head())

# Basic statistics
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nChecking for missing values:")
print(df.isnull().sum())