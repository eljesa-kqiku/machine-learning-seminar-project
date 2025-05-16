import pandas as pd
import numpy as np

df = pd.read_csv('datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')

target = 'Diabetes_binary'
abs_corr_target = df.corr()[target].abs()

low_corr_features = abs_corr_target[(abs_corr_target < 0.1) & (abs_corr_target.index != target)].index.tolist()
df_dropped_low = df.drop(columns=low_corr_features)
print("Low correlation features:", low_corr_features)
print("\nRemaining features after dropping low correlation:", df_dropped_low.columns.tolist())


corr_remaining = df_dropped_low.corr()
upper_triangle = np.triu(np.ones(corr_remaining.shape), k=1).astype(bool)
high_corr_pairs = (
    corr_remaining.where(upper_triangle)
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
filtered_high_corr = high_corr_pairs[high_corr_pairs['Correlation'].abs() > 0.45]
print("\nHigh correlation pairs (> 0.45):")
print(filtered_high_corr.sort_values(by='Correlation', ascending=False))

drop_multicollinearity = set()
abs_corr_remaining_target = df_dropped_low.corr()[target].abs()

for _, row in filtered_high_corr.iterrows():
    f1, f2 = row['Feature 1'], row['Feature 2']
    if f1 != target and f2 != target:
        corr_f1_target = abs_corr_remaining_target.get(f1, 0)
        corr_f2_target = abs_corr_remaining_target.get(f2, 0)
        if corr_f1_target < corr_f2_target and f1 in df_dropped_low.columns:
            drop_multicollinearity.add(f1)
        elif corr_f2_target < corr_f1_target and f2 in df_dropped_low.columns:
            drop_multicollinearity.add(f2)
        elif f2 in df_dropped_low.columns and f2 not in drop_multicollinearity:
            drop_multicollinearity.add(f2)

df_final = df_dropped_low.drop(columns=list(drop_multicollinearity))
print("\nFeatures dropped for multicollinearity:", list(drop_multicollinearity))
print("\nFinal features:", df_final.columns.tolist())


