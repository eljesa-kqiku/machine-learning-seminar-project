from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
df = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

# Separate features and target
X = df.drop(columns=['Diabetes_binary'])  # Features
y = df['Diabetes_binary']                 # Target

# Initialize and apply the scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert scaled features back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Add the target column back
X_scaled_df['Diabetes_binary'] = y.values

# Save to a new CSV file
X_scaled_df.to_csv('diabetes_scaled.csv', index=False)

print("Scaled dataset saved as 'diabetes_scaled.csv'.")
