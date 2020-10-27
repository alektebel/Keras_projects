import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("datasets/IMDB Dataset.csv", encoding='latin-1')

df.sentiment = df.sentiment.replace({"positive": 1, "negative": 0})

dataset = df['review'].to_numpy()
label = df['sentiment'].to_numpy()

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)

tokenizer.fit_on_texts(dataset)

sequences = tokenizer.texts_to_sequences(dataset)

maxlen=np.max([len(i) for i in sequences])

sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=500,padding='pre', truncating='post' )
sequences_train=sequences[0:int(sequences.shape[0]*0.9)]
label_train=label[0:int(sequences.shape[0]*0.9)]
sequences_val=sequences[int(sequences.shape[0]*0.9):]
label_val=label[int(sequences.shape[0]*0.9):]

sequences_train = tf.keras.utils.to_categorical(sequences_train)
sequences_val = tf.keras.utils.to_categorical(sequences_val)

input_shape=sequences_train.shape[1:]

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(5000, 100, input_length=500, input_shape=input_shape),
    tf.keras.layers.GRU(10, activation='relu'),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.05),
    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint('models/imdb_model',
                                                monitor='val_accuracy',
                                                mode='max')
patient =tf.keras.callbacks.EarlyStopping(patience=7)

learning_rate = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch/20))

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(sequences_train)
print(sequences_train[0])
history = model.fit(sequences_train,
          label_train,
          validation_data=(sequences_val, label_val),
          epochs=3,
          batch_size=200,
          verbose=1,
          callbacks=[checkpoint, patient, learning_rate]
          )

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.show()

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

string='I didnt liked the movie, but I think other people, wtih other situation, might like it. I still dont liked it'
seq_pred=tokenizer.texts_to_sequences(string)
sep_pred = tf.keras.preprocessing.sequence.pad_sequences(seq_pred, maxlen=700,padding='pre', truncating='post')

print(model.predict(sep_pred))
