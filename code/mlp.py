import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    mean_squared_error, classification_report
)

# 1. Load data
df = pd.read_csv("datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

# 2. Heatmap e korrelacioneve
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Heatmap - Korrelacionet midis veçorive")
plt.show()

# 3. Features & Labels
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Modeli MLP (në fillim një konfigurim bazik)
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='sgd',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42
)


# 7. Trajnimi
mlp.fit(X_train_scaled, y_train)

# 8. Parashikimi
y_pred = mlp.predict(X_test_scaled)

# 9. Vlerësimi me metrika klasifikimi
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("RMSE:", rmse)

# 10. Confusion matrix vizualisht
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 11. Cross-validation me vizualizim
cv_results = cross_validate(mlp, X_train_scaled, y_train,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1"],
                            return_train_score=True)

# Mesataret
mean_scores = {
    'accuracy': np.mean(cv_results['test_accuracy']),
    'precision': np.mean(cv_results['test_precision']),
    'recall': np.mean(cv_results['test_recall']),
    'f1': np.mean(cv_results['test_f1']),
}

# Bar plot i metrikave nga cross-validation
plt.figure(figsize=(8, 5))
sns.barplot(x=list(mean_scores.keys()), y=list(mean_scores.values()))
plt.title("Metrikat nga Cross Validation (5-Fold)")
plt.ylim(0.5, 1.0)
plt.show()

print("\nKlasifikimi Detajuar:\n")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
