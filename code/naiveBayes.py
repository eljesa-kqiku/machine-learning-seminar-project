import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

# Load training and test sets
X_train = pd.read_csv("datasets/X_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")
y_train = pd.read_csv("datasets/y_train.csv").values.ravel()
y_test = pd.read_csv("datasets/y_test.csv").values.ravel()

# Define and train the Gaussian Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict on the test set
y_pred = gnb.predict(X_test)

# Evaluate performance metrics
print("Performance on Full Feature Set (Naive Bayes):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
disp.plot(cmap=plt.cm.Oranges)
plt.title("Confusion Matrix (Naive Bayes - Full Feature Set)")
plt.show()

# 5-Fold Cross-validation
cv_results = cross_validate(
    gnb, X_train, y_train,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=True
)

print("\nCross-Validation Mean Scores (5-Fold):")
for metric in ["accuracy", "precision", "recall", "f1"]:
    test_score = np.mean(cv_results[f'test_{metric}'])
    train_score = np.mean(cv_results[f'train_{metric}'])
    print(f"{metric.capitalize()}: Train = {train_score:.4f}, Test = {test_score:.4f}")

print("\nDetailed Classification Report (Naive Bayes - Full Feature Set):")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
