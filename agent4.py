import os

import flwr as fl
import tensorflow as tf

from tensorflow import keras
import pandas as pd
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
#model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
#model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# Load and compile Keras model
model = keras.Sequential([
    keras.layers.Input(78,),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=["accuracy"])

# Load dataset
X_webatt_train=pd.read_csv('/client4/X_train.csv')
X_webatt_test=pd.read_csv('/client4/X_test.csv')
y_webatt_train=pd.read_csv('/client4/y_train.csv')
y_webatt_test=pd.read_csv('/client4/y_test.csv')


# Define Flower client
class IDSclient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        for epoch in range(3):
            model.fit(X_webatt_train, y_webatt_train, epochs=1, batch_size=32)
            loss, _, metrics = self.evaluate(model.get_weights(), config)
            print(f"Epoch {epoch+1}: {metrics['accuracy']}% accuracy")
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=IDSclient())
