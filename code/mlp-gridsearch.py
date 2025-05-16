import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load the dataset
df = pd.read_csv("datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# 2. Separate features and target
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Create a pipeline: scaling + MLP
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(max_iter=200, random_state=42))
])

# 5. Define parameter grid for GridSearch
param_grid = {
    'mlp__hidden_layer_sizes': [(100, ), (100, 50)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [0.0001, 0.001],
    'mlp__learning_rate': ['constant', 'adaptive'],
    'mlp__max_iter': [500, 1000]  
}

# 6. Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# 7. Print the best parameters and final evaluation
print("Best Parameters:", grid_search.best_params_)

# 8. Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("Classification Report on Test Set:")
print(classification_report(y_test, y_pred))
