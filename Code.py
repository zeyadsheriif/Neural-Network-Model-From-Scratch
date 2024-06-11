import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

train_X = train_X.reshape(train_X.shape[0], -1) / 255.0
test_X = test_X.reshape(test_X.shape[0], -1) / 255.0

num_classes = 10
train_y = np.eye(num_classes)[train_y]
test_y = np.eye(num_classes)[test_y]

input_size = 784
hidden_size = 128
output_size = 10

weights_1 = np.random.randn(input_size, hidden_size)
weights_2 = np.random.randn(hidden_size, output_size)
biases_1 = np.zeros((1, hidden_size))
biases_2 = np.zeros((1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def feed_forward(x):
    layer_1 = sigmoid(np.dot(x, weights_1) + biases_1)
    layer_2 = sigmoid(np.dot(layer_1, weights_2) + biases_2)
    return layer_2

def backpropagation(x, y):
    global weights_1, weights_2, biases_1, biases_2
    x = x.reshape(1, -1)

    layer_1 = sigmoid(np.dot(x, weights_1) + biases_1)
    layer_2 = sigmoid(np.dot(layer_1, weights_2) + biases_2)
    error = y - layer_2

    delta_2 = error * layer_2 * (1 - layer_2)
    delta_1 = np.dot(delta_2, weights_2.T) * layer_1 * (1 - layer_1)

    weights_2 += np.dot(layer_1.T, delta_2)
    weights_1 += np.dot(x.T, delta_1)
    biases_2 += np.sum(delta_2, axis=0)
    biases_1 += np.sum(delta_1, axis=0)

for epoch in range(10):
    epoch_loss = []
    epoch_accuracy = []
    for i in range(len(train_X)):
        backpropagation(train_X[i], train_y[i])
        prediction = feed_forward(train_X[i])
        epoch_loss.append(np.mean(np.square(train_y[i] - prediction)))
        epoch_accuracy.append(np.argmax(train_y[i]) == np.argmax(prediction))
    val_loss = np.mean(np.square(test_y - feed_forward(test_X)))
    val_accuracy = np.mean(np.argmax(test_y, axis=1) == np.argmax(feed_forward(test_X), axis=1))
    print(f"Epoch {epoch}/{10} - loss: {np.mean(epoch_loss):.4f} - accuracy: {np.mean(epoch_accuracy):.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")

test_loss = np.mean(np.square(test_y - feed_forward(test_X)))
test_accuracy = np.mean(np.argmax(test_y, axis=1) == np.argmax(feed_forward(test_X), axis=1))*100
print("{}/{} - 0s 2ms/step - loss: {:.4f} - accuracy: {:.4f}".format(len(test_X), len(test_X), test_loss, test_accuracy))
print([test_loss, test_accuracy])


y_true = np.argmax(test_y, axis=1)
y_pred = np.argmax(feed_forward(test_X), axis=1)
confusion_matrix = confusion_matrix(y_true, y_pred)
print(confusion_matrix)





