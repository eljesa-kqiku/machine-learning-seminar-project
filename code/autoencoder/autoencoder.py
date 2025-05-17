import numpy as np
import pandas as pd

from keras import layers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

import keras
from keras.models import Sequential
from keras.layers import Input, Dense

X_train = pd.read_csv("datasets/X_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")
y_train = pd.read_csv("datasets/y_train.csv")
y_test = pd.read_csv("datasets/y_test.csv")

input_dim = X_train.shape[1]
encoding_dim = 2

input_layer = keras.layers.Input(shape=(input_dim,))
encoder = keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoder = keras.layers.Dense(input_dim, activation="sigmoid")(encoder)
autoencoder = keras.Model(inputs = input_layer, outputs=decoder)

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.summary()

autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))
encoder = keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[1].output)
encoded_features_train = encoder.predict(X_train)
encoded_features_test = encoder.predict(X_test)

print("Encoded Features Shape (Train):", encoded_features_train.shape)
print("Encoded Features Shape (Test):", encoded_features_test.shape)

# model = LogisticRegression() # 68%
# model = RandomForestClassifier(n_estimators=100, random_state=42) 66%
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42) 74%
# model = SVC(kernel='rbf', probability=True) 69%

model.fit(encoded_features_train, y_train)
y_pred = model.predict(encoded_features_test)
# accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
# print(f"Accuracy with selected features: {accuracy}")