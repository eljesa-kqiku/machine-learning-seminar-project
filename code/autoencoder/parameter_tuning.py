import numpy as np
import pandas as pd
from itertools import product

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense

X_train = pd.read_csv("datasets/X_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")
y_train = pd.read_csv("datasets/y_train.csv")
y_test = pd.read_csv("datasets/y_test.csv")

class Autoencoder:
    def __init__(self, encoding_dim, activation1, activation2, optimizer1, optimizer2):
        self.input_dim = X_train.shape[1]
        self.encoding_dim = encoding_dim

        # Build the autoencoder model
        input_layer = keras.layers.Input(shape=(self.input_dim,))
        encoder = keras.layers.Dense(encoding_dim, activation1)(input_layer)
        decoder = keras.layers.Dense(self.input_dim, activation2)(encoder)

        self.autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
        self.encoder_model = keras.Model(inputs=input_layer, outputs=encoder)

        self.autoencoder.compile(optimizer=optimizer1, loss=optimizer2)
        self.autoencoder.summary()

    def train(self, X_train, X_test, epochs, batch_size=32):
        self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, X_test)
        )

    def encode(self, X):
        return self.encoder_model.predict(X)

ae_param_map = {
    'activation1': ['leaky_relu'],
    'activation2': ['relu'],
    'optimizer1': ['adam', 'sgd'],
    'optimizer2': ['mse', 'binary_crossentropy']
}
gb_learning_rates = [0.1, 0.3]

ae_keys = list(ae_param_map.keys())
ae_values = list(ae_param_map.values())
ae_combinations = list(product(*ae_values))

results = []
for ae_params_tuple in ae_combinations:
    ae_params = dict(zip(ae_keys, ae_params_tuple))
    print(f"Training Autoencoder with params: {ae_params}")

    ae = Autoencoder(
        encoding_dim=4,
        activation1=ae_params['activation1'],
        activation2=ae_params['activation2'],
        optimizer1=ae_params['optimizer1'],
        optimizer2=ae_params['optimizer2']
    )
    ae.train(X_train, X_test, epochs=50)
    encoded_train = ae.encode(X_train)
    encoded_test = ae.encode(X_test)

    for lr in gb_learning_rates:
        print(f"Training GradientBoosting with learning_rate={lr}")
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=lr,
            random_state=42
        )
        model.fit(encoded_train, y_train.values.ravel())

        y_pred = model.predict(encoded_test)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='macro')

        print(f"Accuracy: {acc:.4f}, Recall: {rec:.4f}\n")

        results.append({
            'activation1': ae_params['activation1'],
            'activation2': ae_params['activation2'],
            'optimizer1': ae_params['optimizer1'],
            'optimizer2': ae_params['optimizer2'],
            'gb_learning_rate': lr,
            'accuracy': acc,
            'recall_macro': rec
        })

results_df = pd.DataFrame(results)

print("Tuning completed. Summary:")
print(results_df)