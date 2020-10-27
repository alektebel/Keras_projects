import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(series, dtype=tf.float64))
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)

    return forecast

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = np.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(series, dtype=tf.float64))
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    plt.show()

def plot_this(time, series):
    plt.plot(time, series)
    plt.xlabel("time")
    plt.ylabel("Value")
    plt.grid(True)



df = pd.read_csv('Sunspots.csv')
dataset = df.to_numpy()
series=dataset[:,2]
time=dataset[:,0]





split_time = 2500
window_size = 30
batch_size=200


time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]


data = windowed_dataset(x_train, window_size, batch_size, 1000)
print(data)
print(x_train.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 600)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

patient = tf.keras.callbacks.EarlyStopping(patience=2,
                                           monitor='mae',
                                           mode='min'
                                           )

optimizer = tf.keras.optimizers.Adam(lr=1e-8)

model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])



history = model.fit(data, epochs=150, callbacks=[lr_schedule, patient])

rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)

print(rnn_forecast.shape)
rnn_forecast = rnn_forecast[(split_time - window_size):-1, -1]



plt.figure(figsize=(10, 6))
plot_this(time_valid, x_valid)
plot_this(time_valid, rnn_forecast)
plt.show()