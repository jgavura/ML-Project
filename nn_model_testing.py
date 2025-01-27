import numpy as np
from tensorflow import keras


# loading a trained model
model = keras.models.load_model("model.keras")

# loading normalization parameters
with open('model_mean_std.txt', 'r') as f:
    data = f.read().split('\n')[0].split(' ')
    x_mean = list(map(float, data[0].split(',')))
    x_std = list(map(float, data[1].split(',')))
    y_mean = float(data[2])
    y_std = float(data[3])


# loading testing data
x_test, y_test = [], []

with open('data/testing_set.txt', 'r') as f:
    for line in f.read().split('\n'):
        if line == '':
            continue
        x, y, z = map(float, line.split(' '))
        x_test.append((x, y))
        y_test.append(z)

x_test, y_test = np.array(x_test), np.array(y_test)


# normalize testing data
x_test_standardized = (x_test - x_mean) / x_std
y_test_standardized = (y_test - y_mean) / y_std


# Testing the model
test_score = model.evaluate(x_test_standardized, y_test_standardized, verbose=0)
print("\nNN test loss: {} | NN test mae: {}".format(test_score[0], test_score[1]))


# saving predictions for testing set
predictions_standardized = model.predict(x_test_standardized, verbose=0)
predictions = predictions_standardized * y_std + y_mean

# with open('data/testing_set_nn_predictions.txt', 'w') as f:
#     for i in range(x_test.shape[0]):
#         f.write(f'{x_test[i][0]} {x_test[i][1]} {predictions[i][0]}\n')
