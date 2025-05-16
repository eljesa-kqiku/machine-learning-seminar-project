import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# 1. Load data
df = pd.read_csv("datasets/diabetes_scaled.csv")

# 2. Features & Labels
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# 3. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Define MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='sgd',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)

# 5. Train model
mlp.fit(X_train, y_train)

# 6. Predict on test set
y_pred = mlp.predict(X_test)

# Plot loss curve for full feature set
plt.figure(figsize=(8, 4))
plt.plot(mlp.loss_curve_)
plt.title("Loss Curve (Full Feature Set)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 7. Evaluate metrics
print("Performance on Full Feature Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# 8. Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix (Full Feature Set)")
plt.show()

# 9. Cross-validation without plot
cv_results = cross_validate(
    mlp, X_train, y_train,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=True
)

print("\nCross-Validation Mean Scores (5-Fold):")
for metric in ["accuracy", "precision", "recall", "f1"]:
    test_score = np.mean(cv_results[f'test_{metric}'])
    train_score = np.mean(cv_results[f'train_{metric}'])
    print(f"{metric.capitalize()}: Train = {train_score:.4f}, Test = {test_score:.4f}")

print("\nDetailed Classification Report (Full Feature Set):")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

# === Performance on Selected Features + Interaction Terms ===

# Select features of interest + target
df_interact = df[["GenHlth", "PhysHlth", "DiffWalk", "Education", "Income", "Diabetes_binary"]].copy()

# Add interaction features
df_interact["GenHlth_x_PhysHlth"] = df_interact["GenHlth"] * df_interact["PhysHlth"]
df_interact["PhysHlth_x_DiffWalk"] = df_interact["PhysHlth"] * df_interact["DiffWalk"]
df_interact["GenHlth_x_DiffWalk"] = df_interact["GenHlth"] * df_interact["DiffWalk"]
df_interact["Education_x_Income"] = df_interact["Education"] * df_interact["Income"]

# Prepare features and labels
X_interact = df_interact.drop("Diabetes_binary", axis=1)
y_interact = df_interact["Diabetes_binary"]

# Train/Test split
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_interact, y_interact, test_size=0.2, stratify=y_interact, random_state=42
)

# Define MLP model
mlp_interact = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='sgd',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)

# Train model
mlp_interact.fit(X_train_i, y_train_i)

# Predict on test set
y_pred_i = mlp_interact.predict(X_test_i)

# Plot loss curve for selected features + interaction terms
plt.figure(figsize=(8, 4))
plt.plot(mlp_interact.loss_curve_)
plt.title("Loss Curve (Interactions Model)")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Evaluate metrics
print("\nPerformance on Selected Features + Interactions:")
print(f"Accuracy: {accuracy_score(y_test_i, y_pred_i):.4f}")
print(f"Precision: {precision_score(y_test_i, y_pred_i):.4f}")
print(f"Recall: {recall_score(y_test_i, y_pred_i):.4f}")
print(f"F1 Score: {f1_score(y_test_i, y_pred_i):.4f}")

# Confusion matrix visualization
cm_i = confusion_matrix(y_test_i, y_pred_i)
disp_i = ConfusionMatrixDisplay(confusion_matrix=cm_i, display_labels=mlp_interact.classes_)
disp_i.plot(cmap=plt.cm.Greens)
plt.title("Confusion Matrix (Interactions Model)")
plt.show()

print("\nDetailed Classification Report (Interactions Model):")
print(classification_report(y_test_i, y_pred_i, target_names=["No Diabetes", "Diabetes"]))
