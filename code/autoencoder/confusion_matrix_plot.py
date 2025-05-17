import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Confusion matrix values
conf_matrix = np.array([[400, 109],
                        [163, 328]])

# Create a labeled DataFrame
labels = ['Actual Positive', 'Actual Negative']
columns = ['Predicted Positive', 'Predicted Negative']
df_cm = pd.DataFrame(conf_matrix, index=labels, columns=columns)

# Plot
plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()

plt.savefig("report/images/autoencoder_confusion_matrix.png", dpi=300)