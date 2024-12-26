import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Suppress OpenMP duplicate error
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend to prevent conflicts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from datetime import datetime

start = datetime.now()

# Load the dataset (same as in dnn_train.py)
df = pd.read_csv('dataset_norm.csv', header=None)
data = np.array(df)

# Load dates from 'sentinel_values_predict_dates.csv'
dates_df = pd.read_csv('sentinel_values_predict_dates.csv', header=None)
dates = dates_df.iloc[:, 0]  # Assuming the dates are in the first column
dates = pd.to_datetime(dates, dayfirst=True)

# Check that the number of dates matches the number of samples
assert len(dates) == data.shape[0], "Number of dates does not match number of samples"

# Split data into input features and target variables (same as in dnn_train.py)
x_set = data[:, :-4]  # First values as input
y_set = data[:, -4:]  # Last 4 values as target

print("Input shape:", x_set.shape)
print("Target shape:", y_set.shape)

print("First input sample:", x_set[0])
print("First target sample:", y_set[0])

# Ensure data is in float32 format
x_set = x_set.astype('float32')
y_set = y_set.astype('float32')

# Titles for outputs
titles_for_outputs = ['Water temperature', 'Electrical Conductivity', 'Dissolved Oxygen', 'Chemical Oxygen Demand']

# Define the model architecture (same as in dnn_train.py)
def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,)))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.01))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(output_dim, activation='sigmoid'))
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    model.summary()
    return model

# Create the model
model = create_model(input_dim=x_set.shape[1], output_dim=y_set.shape[1])

# Load the weights
model_path = 'dnn_model.h5'
print('Loading weights from:', model_path)
model.load_weights(model_path)

# Function to de-normalize predictions and targets
def de_normalize(np_line, np_min, np_max):
    for x in range(np_line.shape[0]):
        for y in range(np_line.shape[1]):
            xmin = np_min[0, -(4 - y)]
            xmax = np_max[0, -(4 - y)]
            np_line[x, y] = np.round((np_line[x, y]) * (xmax - xmin) + xmin, 6)
    return np_line

# Function to plot and compare expected vs predicted values
def plot_predictions(outputs, predictions, dates):
    """
    Plot and compare expected vs predicted values for each parameter.
    """
    num_params = outputs.shape[1]
    for i in range(num_params):
        plt.figure(figsize=(12, 6))
        plt.plot(dates, outputs[:, i], label='Expected Values', marker='o', linestyle='--', alpha=0.7)
        plt.plot(dates, predictions[:, i], label='Predicted Values', marker='x', linestyle='-', alpha=0.7)
        plt.title(f"DNN - {titles_for_outputs[i]}: Expected vs. Predicted")
        plt.xlabel("Date")
        plt.ylabel(f"{titles_for_outputs[i]} Value")
        plt.legend()
        plt.grid()
        # Format the x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Rotate date labels automatically
        plt.savefig(f'dnn_parameter_{i+1}_comparison.png', dpi=300)
        plt.close()
    print("Graphs for each parameter have been saved as PNG files.")

# Load min and max values for de-normalization
print('Loading min and max values for de-normalization...')
pd_min = pd.read_csv('min.csv', header=None)
np_min = np.array(pd_min)
pd_max = pd.read_csv('max.csv', header=None)
np_max = np.array(pd_max)

# Predict on the entire dataset
print('Making predictions on the entire dataset...')
preds_all = model.predict(x_set, verbose=1)
model_acc = model.evaluate(x_set, y_set, verbose=1)
print("Predictions shape:", preds_all.shape)
print("Input shape:", x_set.shape)
print("Target shape:", y_set.shape)

# Calculate evaluation metrics
mape_loss = tf.keras.metrics.mean_absolute_percentage_error(y_set, preds_all)
mse_loss = tf.keras.metrics.mean_squared_error(y_set, preds_all)
mae_loss = tf.keras.metrics.mean_absolute_error(y_set, preds_all)
rmse_loss = tf.sqrt(mse_loss)

print('Model evaluation on the entire dataset:')
print('Mean Absolute Percentage Error (MAPE):', float(tf.reduce_mean(mape_loss)))
print('Mean Absolute Error (MAE):', float(tf.reduce_mean(mae_loss)))
print('Root Mean Squared Error (RMSE):', float(tf.reduce_mean(rmse_loss)))
print('Mean Squared Error (MSE):', float(tf.reduce_mean(mse_loss)))

# De-normalize predictions and targets
print('De-normalizing predicted and target values...')
preds_all_denorm = de_normalize(preds_all.copy(), np_min, np_max)
y_set_denorm = de_normalize(y_set.copy(), np_min, np_max)

# Plot and compare expected vs predicted values
print('Plotting predictions...')
plot_predictions(y_set_denorm, preds_all_denorm, dates)

# Write predictions to a CSV file
print('Writing predictions to CSV file...')
with open('dnn_predictions_all.csv', 'w') as f:
    f.write('Date,Expected Values,Predicted Values\n')
    for i in range(preds_all_denorm.shape[0]):
        date_str = dates.iloc[i].strftime('%Y-%m-%d')
        expected_values = ','.join(map(str, y_set_denorm[i]))
        predicted_values = ','.join(map(str, preds_all_denorm[i]))
        f.write(f"{date_str},{expected_values},{predicted_values}\n")

finish_time = datetime.now()
print('Total prediction time:', finish_time - start)
