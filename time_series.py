import tensorflow as tf
#series means the list of values in the dataset: could be the prices of a stock, the temperature of a room hour by hour...
#the window is the range of values we want to take on each example. the batch size is the number of examples we take each time to train
#the optimizer. the shuffle
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset