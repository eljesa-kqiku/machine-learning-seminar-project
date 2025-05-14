import pandas as pd

# Load the dataset
df = pd.read_csv('datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Check for missing values
missing_values = df.isnull().sum()

# Display missing values per column
print(missing_values)
