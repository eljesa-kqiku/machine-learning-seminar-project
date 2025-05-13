import pandas as pd

# Load the dataset
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Check for duplicate rows
duplicates = df[df.duplicated()]

# Display the result
if duplicates.empty:
    print("No duplicates found!")
else:
    print(f"Found {duplicates.shape[0]} duplicate rows.")
    print(duplicates)
