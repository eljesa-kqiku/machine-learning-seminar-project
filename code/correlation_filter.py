import pandas as pd
import numpy as np

df = pd.read_csv('datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
corr_matrix = df.corr()

# Unstack and filter correlations
high_corr = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr.columns = ['Feature 1', 'Feature 2', 'Correlation']
filtered_corr = high_corr[high_corr['Correlation'].abs() > 0.45]
print(filtered_corr.sort_values(by='Correlation', ascending=False))
