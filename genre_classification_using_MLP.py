import  json
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow.keras as keras

DATA_PATH = "/Users/suyashramteke/PycharmProjects/Audio_DL/Data/Data.json"

def load_data(data_path):

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data['labels'])

    print("Data successfully loaded")

    return X, y

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

if __name__ == "__main__" :
    #Load data
    X, y = load_data(DATA_PATH)
    #Create test and training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    print(X.shape)
    #Build the network
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (X.shape[1], X.shape[2])),
        keras.layers.Dense(512, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.002)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(256, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.002)),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128, activation = 'relu', kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),


        keras.layers.Dense(10, activation = 'softmax')
        ])

    #compile model
    optimizer= keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

    #train model
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), batch_size = 32, epochs = 90)

    plot_history(history)








