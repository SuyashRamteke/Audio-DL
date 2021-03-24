import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from random import random

# Generate Dataset

def generate_dataset(num_samples, test_size = 0.3):
    x = np.array([[random()/2 for _ in range(2)] for _ in range(num_samples)])
    y = np.array([[i[0] + i[1]] for i in x])

    #split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':

    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.33)

    # Build Model
    model = tf.keras.Sequential([tf.keras.layers.Dense(7, input_dim=2, activation = "sigmoid"),
                                tf.keras.layers.Dense(1, activation = "sigmoid")])

    # Stochastic Gradient Descent
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1)

    # Compile model
    model.compile(optimizer = optimizer, loss = "mse")

    # Train model
    model.fit(x_train, y_train, epochs = 100)

    # Test Model
    model.evaluate(x_test, y_test, verbose = 2)

    # get predictions
    data = np.array([[0.1, 0.2], [0.2, 0.2]])
    predictions = model.predict(data)

    # print predictions
    print("\nPredictions:")
    for d, p in zip(data, predictions):
        print("{} + {} = {}".format(d[0], d[1], p[0]))



