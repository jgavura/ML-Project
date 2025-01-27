import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt


TF_ENABLE_ONEDNN_OPTS=0


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


# creating a nn model
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])

# compiling the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

# setting up early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

# training the model
history = model.fit(
    x_train_standardized, y_train_standardized,
    validation_data=(x_val_standardized, y_val_standardized),
    epochs=200,
    batch_size=16,
    verbose=1,
    callbacks=[early_stopping]
)


# plot losses over epochs
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend(loc='best')

plt.tight_layout()
plt.show()


# evaluate trained model
train_score = model.evaluate(x_train_standardized, y_train_standardized, verbose=0)
print("\nTrain loss: {} | Train mae: {}".format(train_score[0], train_score[1]))

val_score = model.evaluate(x_val_standardized, y_val_standardized, verbose=0)
print("\nValidation loss: {} | Validation mae: {}".format(val_score[0], val_score[1]))


# saving the model
# model.save("model.keras")

# saving normalization parameters
# with open('model_mean_std.txt', 'w') as f:
#     f.write(f'{x_mean[0]},{x_mean[1]} {x_std[0]},{x_std[1]} {y_mean} {y_std}')

