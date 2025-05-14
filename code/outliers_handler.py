import pandas as pd
import numpy as np
from scipy.stats import zscore

df = pd.read_csv('datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

####### Making a summary of the fields #######
min_vals = df.min()
max_vals = df.max()
avg_vals = df.mean()
mode_vals = df.mode().iloc[0]
summary_df = pd.DataFrame({
    'Min': min_vals,
    'Max': max_vals,
    'Mean': avg_vals,
    'Mode': mode_vals
})
print("####### Summary of the dataframe #######")
print(summary_df)


print("\n####### Checking for outliers with Interquartile Range (IQR) method #######")
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
outlier_counts = outliers.sum(axis=1)
print("Rows with >=1 outliers:", (outlier_counts >= 1).sum())


print("\n####### Checking for outliers with Z-Score Method (Assumes Normal Distribution) #######")
z_scores = df.select_dtypes(include='number').apply(zscore)
outliers = (z_scores.abs() > 5)
print("Outlier rows (Z-score):", outliers.any(axis=1).sum())


####### Removing the outliers #######
rows_to_remove = outliers.any(axis=1)
cleaned_df = df[~rows_to_remove]
cleaned_df.to_csv('datasets/dataset_without_outliers.csv', index=False)
print("Cleaned dataset saved as 'dataset_without_outliers.csv'")
