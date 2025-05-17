import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/dafasetF.csv')
X = df.drop(columns=['Diabetes_binary'])
y = df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_train.to_csv('datasets/X_train.csv', index=False)
X_test.to_csv('datasets/X_test.csv', index=False)
y_train.to_csv('datasets/y_train.csv', index=False)
y_test.to_csv('datasets/y_test.csv', index=False)

print("Data successfully split and saved.")
