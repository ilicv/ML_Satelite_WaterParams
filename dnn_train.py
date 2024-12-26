import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from datetime import datetime
import keras

start = datetime.now()

# Load and prepare the data
df = pd.read_csv('dataset_norm.csv', header=None)
data = np.array(df)

x_set = data[:, :-4]  # First values as input
y_set = data[:, -4:]  # Last 4 values as target

print("Input shape:", x_set.shape)
print("Target shape:", y_set.shape)

print("First input sample:", x_set[0])
print("First target sample:", y_set[0])

# Split the data
x_train, x_temp, y_train, y_temp = train_test_split(x_set, y_set, test_size=0.25, random_state=0)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=0)

print("Training input shape:", x_train.shape)
print("Validation input shape:", x_val.shape)
print("Test input shape:", x_test.shape)
print("Training target shape:", y_train.shape)
print("Validation target shape:", y_val.shape)
print("Test target shape:", y_test.shape)

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.01))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(y_set.shape[1], activation='sigmoid'))
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    model.summary()
    return model

nnet = create_model(input_dim=x_set.shape[1])

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_mse', patience=45, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor='val_mse', factor=0.01, patience=35, min_lr=0.00001, verbose=1),
    ModelCheckpoint('dnn_model.h5', monitor='val_mse', verbose=1, save_best_only=True, save_weights_only=True)
]

print('Training model...')
model_history = nnet.fit(
    x_train, y_train,
    batch_size=1,
    epochs=300,
    shuffle=True,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print('Loading best model...')
nnet.load_weights('dnn_model.h5')

def plot_losses(train_losses, val_losses):
    """
    Plot training and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", marker="o", linestyle="--", alpha=0.7)
    plt.plot(val_losses, label="Validation Loss", marker="x", linestyle="-", alpha=0.7)
    plt.title("DNN - Training and Validation Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid()
    plt.savefig("DNN_training_validation_loss.png", dpi=300)
    plt.show()
    print("DNN_Training and validation loss graph has been saved as 'training_validation_loss.png'.")

# Visualizing training and validation loss
history_dict = model_history.history
train_losses = history_dict['mse']
val_losses = history_dict['val_mse']

plot_losses(train_losses, val_losses)

print('Predicting test set values...')
preds_test = nnet.predict(x_test, verbose=1)
model_acc = nnet.evaluate(x_test, y_test)
print("Test set predictions shape:", preds_test.shape)
print("Test set input shape:", x_test.shape)
print("Test set target shape:", y_test.shape)

mape_loss = keras.metrics.mean_absolute_percentage_error(y_test, preds_test)
mse_loss = keras.metrics.mean_squared_error(y_test, preds_test)

def de_normalize(np_line, np_min, np_max):
    for x in range(np_line.shape[0]):
        for y in range(np_line[x].shape[0]):
            xmin = np_min[0, -(4 - y)]
            xmax = np_max[0, -(4 - y)]
            np_line[x, y] = np.round((np_line[x, y]) * (xmax - xmin) + xmin, 6)
    return np_line

print('Loading min and max values for denormalization...')
pd_min = pd.read_csv('min.csv', header=None)
np_min = np.array(pd_min)
pd_max = pd.read_csv('max.csv', header=None)
np_max = np.array(pd_max)

print('Denormalizing predicted values...')
preds_test = de_normalize(preds_test, np_min, np_max)
y_test = de_normalize(y_test, np_min, np_max)

print('Writing predictions to CSV file...')
with open('predictions_test.csv', 'w') as f:
    f.write('expected, predicted\n')
    for x in range(preds_test.shape[0]):
        expe = ','.join(map(str, y_test[x]))
        pred = ','.join(map(str, preds_test[x]))
        f.write(f"{expe},{pred}\n")

print('Model accuracy:', model_acc)
print('Mean Absolute Percentage Error (MAPE):', float(tf.reduce_mean(mape_loss)))
print('Mean Squared Error (MSE):', float(tf.reduce_mean(mse_loss)))

finish_time = datetime.now()
print('Total training time:', finish_time - start)

# Additional evaluations on the entire dataset and training set can be added here if needed
