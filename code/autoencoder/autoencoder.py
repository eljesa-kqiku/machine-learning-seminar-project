import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# importing the dataset
X_train = pd.read_csv("datasets/X_train.csv")
X_test = pd.read_csv("datasets/X_test.csv")
y_train = pd.read_csv("datasets/y_train.csv")
y_test = pd.read_csv("datasets/y_test.csv")

class Autoencoder:
    def __init__(self, encoding_dim, activation1, activation2, optimizer1, optimizer2):
        self.input_dim = X_train.shape[1]
        self.encoding_dim = encoding_dim

        # constructing the layers
        input_layer = keras.layers.Input(shape=(self.input_dim,))
        encoder = keras.layers.Dense(encoding_dim, activation1)(input_layer)
        decoder = keras.layers.Dense(self.input_dim, activation2)(encoder)

        # building the encoder and decoder
        self.autoencoder = keras.Model(inputs=input_layer, outputs=decoder)
        self.encoder_model = keras.Model(inputs=input_layer, outputs=encoder)

        self.autoencoder.compile(optimizer=optimizer1, loss=optimizer2)
        self.autoencoder.summary()

    def train(self, X_train, X_test, epochs, batch_size=32):
        # training the model to learn from dataset
        self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(X_test, X_test)
        )

    def encode(self, X):
        return self.encoder_model.predict(X)

# initializing the autoencoder
ae = Autoencoder(
    encoding_dim=4,
    activation1='leaky_relu',
    activation2='sigmoid',
    optimizer1='adam',
    optimizer2='mse'
)

# running the training steps for feature extraction
ae.train(X_train, X_test, epochs=50)
encoded_train = ae.encode(X_train)
encoded_test = ae.encode(X_test)

# applying the classifier
model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.3,
    random_state=42
)

model.fit(encoded_train, y_train.values.ravel())
y_pred = model.predict(encoded_test)

#printing the classification_report
report = classification_report(y_test, y_pred)
print(report)