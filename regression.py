import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def expand_features(x, dim):
    x1 = x[:, 1]
    x2 = x[:, 2]

    expanded_x = np.hstack((x, (x1 * x2)[:, np.newaxis]))

    for i in range(2, dim+1):
        expanded_x = np.hstack((expanded_x, (x1 ** i)[:, np.newaxis], (x2 ** i)[:, np.newaxis]))

    print(expanded_x.shape)

    return expanded_x


# reading data from text files
x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []

with open('data/training_set.txt', 'r') as f:
    for line in f.read().split('\n'):
        if line == '':
            continue
        x, y, z = map(float, line.split(' '))
        x_train.append((x, y))
        y_train.append(z)

with open('data/validation_set.txt', 'r') as f:
    for line in f.read().split('\n'):
        if line == '':
            continue
        x, y, z = map(float, line.split(' '))
        x_val.append((x, y))
        y_val.append(z)

with open('data/testing_set.txt', 'r') as f:
    for line in f.read().split('\n'):
        if line == '':
            continue
        x, y, z = map(float, line.split(' '))
        x_test.append((x, y))
        y_test.append(z)

x_train, y_train = np.array(x_train), np.array(y_train)
x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)


# normalizing data by standardization
x_all = np.concatenate((x_train, x_val, x_test))
y_all = np.concatenate((y_train, y_val, y_test))

x_mean = x_all.mean(axis=0)
x_std = x_all.std(axis=0)
y_mean = y_all.mean()
y_std = y_all.std()

x_train_standardized = (x_train - x_mean) / x_std
y_train_standardized = (y_train - y_mean) / y_std

x_val_standardized = (x_val - x_mean) / x_std
y_val_standardized = (y_val - y_mean) / y_std

x_test_standardized = (x_test - x_mean) / x_std
y_test_standardized = (y_test - y_mean) / y_std


# adding the intercept
x_train_standardized = np.insert(x_train_standardized, 0, 1, axis=1)
x_val_standardized = np.insert(x_val_standardized, 0, 1, axis=1)
x_test_standardized = np.insert(x_test_standardized, 0, 1, axis=1)


# closed form regression
closed_form_train_x = np.concatenate((x_train_standardized, x_val_standardized))
closed_form_train_y = np.concatenate((y_train_standardized, y_val_standardized))
closed_form_theta = np.linalg.inv(closed_form_train_x.T.dot(closed_form_train_x)).dot(closed_form_train_x.T).dot(closed_form_train_y)


# evaluating on testing set
predictions_standardized = x_test_standardized.dot(closed_form_theta)
mse = mean_squared_error(y_test_standardized, predictions_standardized)
mae = mean_absolute_error(y_test_standardized, predictions_standardized)
print("\nClosed form reg test loss: {} | Closed form reg test mae: {}".format(mse, mae))


# save closed form predictions
predictions = predictions_standardized * y_std + y_mean
# with open('data/testing_set_closed_form_reg_predictions.txt', 'w') as f:
#     for i in range(x_test.shape[0]):
#         f.write(f'{x_test[i][0]} {x_test[i][1]} {predictions[i]}\n')


# expanding the data set
expanded_train_x = expand_features(x_train_standardized, 3)
expanded_val_x = expand_features(x_val_standardized, 3)
expanded_test_x = expand_features(x_test_standardized, 3)


# generalized closed form regression
generalized_closed_form_theta = np.linalg.inv(expanded_train_x.T.dot(expanded_train_x)).dot(expanded_train_x.T).dot(y_train_standardized)

# evaluating on training set
predictions_standardized = expanded_train_x.dot(generalized_closed_form_theta)
mse = mean_squared_error(y_train_standardized, predictions_standardized)
mae = mean_absolute_error(y_train_standardized, predictions_standardized)
print("\nGeneralized closed form 3 train loss: {} | Generalized closed form 3 train mae: {}".format(mse, mae))

# evaluating on validation set
predictions_standardized = expanded_val_x.dot(generalized_closed_form_theta)
mse = mean_squared_error(y_val_standardized, predictions_standardized)
mae = mean_absolute_error(y_val_standardized, predictions_standardized)
print("\nGeneralized closed form 3 val loss: {} | Generalized closed form 3 val mae: {}".format(mse, mae))

# evaluating on testing set
predictions_standardized = expanded_test_x.dot(generalized_closed_form_theta)
mse = mean_squared_error(y_test_standardized, predictions_standardized)
mae = mean_absolute_error(y_test_standardized, predictions_standardized)
print("\nGeneralized closed form 3 test loss: {} | Generalized closed form 3 test mae: {}".format(mse, mae))


# save generalized closed form predictions
predictions = predictions_standardized * y_std + y_mean
# with open('data/testing_set_generalized_closed_form_reg_predictions.txt', 'w') as f:
#     for i in range(x_test.shape[0]):
#         f.write(f'{x_test[i][0]} {x_test[i][1]} {predictions[i]}\n')

