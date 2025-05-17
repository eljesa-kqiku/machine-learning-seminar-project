ae = Autoencoder(
        encoding_dim=4,
        activation1='leaky_relu',
        activation2='sigmoid',
        optimizer1='adam',
        optimizer2='mse'
    )

ae.train(X_train, X_test, epochs=50)
encoded_train = ae.encode(X_train)
encoded_test = ae.encode(X_test)

model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.3,
            random_state=42
        )

model.fit(encoded_train, y_train.values.ravel())
y_pred = model.predict(encoded_test)

report = classification_report(y_test, y_pred)
print(report)