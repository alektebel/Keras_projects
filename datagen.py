import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.text import Tokenizer


def generator(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield (features[n*batch_size:(n+1)*batch_size], labels[n*batch_size:(n+1)*batch_size])

def string2onehot(x):
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(train_labels)

    seq = tokenizer.texts_to_sequences(train_labels)
    seq = np.array(seq)[:, 1]
    y = tf.keras.utils.to_categorical(seq, 5)[:,1:]
    y = y[:, 1:]
    return y

dataset = pd.read_csv('Iris.csv')
dataset=shuffle(dataset)
data= dataset.to_numpy()

train_data=data[0:int(data.shape[0]*0.9),1:4]
train_labels=data[0:int(data.shape[0]*0.9),5]
test_data=data[int(data.shape[0]*0.9):,1:4]
test_labels=data[int(data.shape[0]*0.9):,5]
print(train_labels)
print(test_labels)
print(string2onehot(test_labels))
tokenizer = Tokenizer()

tokenizer.fit_on_texts(train_labels)

seq = tokenizer.texts_to_sequences(train_labels)
seq=np.array(seq)[:,1]
y=tf.keras.utils.to_categorical(seq,num_classes=5)[:,1:]

y=y[:,1:]

y_test = string2onehot(test_labels)



train_data = np.asarray(train_data).astype(np.float32)
test_data = np.asarray(test_data).astype(np.float32)


input_shape = (4,)

inputs=tf.keras.layers.Input(input_shape)
x=tf.keras.layers.Dense(10, activation='relu')(inputs)
output=tf.keras.layers.Dense(4, activation='softmax')(x)

model = tf.keras.Model([inputs], output)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
training_features = train_data
training_labels = y
validation_features = test_data
validation_labels = y_test



epochs=3
batch_size = 20
train_steps=len(train_data) // batch_size
for i in range(epochs):
    train_datagen = generator(training_features, training_labels, batch_size=batch_size)
    validation_datagen = generator(validation_features, validation_labels, batch_size=batch_size)
    model.fit_generator(train_datagen, steps_per_epoch=train_steps, validation_data = validation_datagen, validation_steps=1)


