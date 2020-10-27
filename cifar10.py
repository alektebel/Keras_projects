import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=tf.keras.datasets.cifar100.load_data()

(X, labels), (X_test, labels_test) = dataset

y = tf.keras.utils.to_categorical(labels)
y_test = tf.keras.utils.to_categorical(labels_test)

model_in = tf.keras.applications.ResNet101V2(include_top=False, input_shape=(32, 32, 3))
model_in.trainable=False
model=tf.keras.Sequential([
    model_in,
    tf.keras.layers.Dense(100, activation='softmax')
])
model_new=tf.keras.Sequential([
    tf.keras.layers.Conv2D(50, (3,3), kernel_regularizer=tf.keras.regularizers.l1(), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(100, (3,3), kernel_regularizer=tf.keras.regularizers.l1(),activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='softmax')
])
model_new.summary()
model.summary()

model_new.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y=np.expand_dims(y, axis=1)
y=np.expand_dims(y, axis=1)
y_test=np.expand_dims(y_test, axis=1)
y_test=np.expand_dims(y_test, axis=1)

history = model.fit(X,
          y,
          epochs=5,
          batch_size=256,
          validation_data=(X_test, y_test))

history_new = model.fit(X,
          y,
          epochs=5,
          batch_size=256,
          validation_data=(X_test, y_test))

def plot_history(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.title('epochs vs '+string)
    plt.ylabel(string)
    plt.xlabel('epochs')
    plt.legend([string, 'val_'+string])
    plt.show()

plot_history(history, 'accuracy')
plot_history(history_new, 'accuracy')