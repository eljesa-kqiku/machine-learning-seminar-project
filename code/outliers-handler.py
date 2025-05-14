import pandas as pd
import numpy as np

df = pd.read_csv('datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

####### Making a summary of the fields #######
# min_vals = df.min()
# max_vals = df.max()
# avg_vals = df.mean()
# mode_vals = df.mode().iloc[0]  # Take the first mode if there are multiple
#
# # Combine them into a single DataFrame for easier readability
# summary_df = pd.DataFrame({
#     'Min': min_vals,
#     'Max': max_vals,
#     'Mean': avg_vals,
#     'Mode': mode_vals
# })
# print(summary_df)

####### Checking for outliers with z-score method #######


# Q1 = df.quantile(0.25)
# Q3 = df.quantile(0.75)
# IQR = Q3 - Q1
#
# outliers = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))

# # Rows with at least 2 outlier values
# outlier_counts = outliers.sum(axis=1)
# print("Rows with >1 outliers:", (outlier_counts >= 1).sum())



#### qe kjo mi ka jep 1905
# from scipy.stats import zscore
#
# z_scores = df.select_dtypes(include='number').apply(zscore)
# outliers = (z_scores.abs() > 5)
#
# # Count rows with any outlier
# print("Outlier rows (Z-score):", outliers.any(axis=1).sum())

import numpy as np

log_df = df.select_dtypes(include='number').apply(lambda x: np.log1p(x))

z_scores = log_df.apply(z_scores)
outliers = z_scores.abs() > 3

print("Outlier rows (log-transformed Z-score):", outliers.any(axis=1).sum())