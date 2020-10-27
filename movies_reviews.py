import tensorflow as tf
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def list2string(list):
    r=" "
    for i in list:
        if type(i) is float:
            continue
        r=r+i+" "
    return r


df = pd.read_csv("movie/movies_metadata.csv", low_memory=False)

data=df[['overview',
         'title',
          'adult',
         'budget',
         'runtime',
         'vote_average',
         'vote_count',
         'popularity']]
#data["adult"]=data["adult"].replace{ True:1, False:0}

dataset = data.to_numpy()

np.random.shuffle(dataset)

train_data = dataset[:,:-1]
label_data = dataset[:,-1]



print(data.columns)


data['title'].astype(str)
data['overview'].astype(str)
training_data = train_data[0:int(train_data.shape[0]*0.9), :]
training_labels = label_data[0:int(train_data.shape[0]*0.9)]
validation_data = train_data[int(train_data.shape[0]*0.9):, :]
validation_labels = label_data[int(train_data.shape[0]*0.9):]

title_token = tf.keras.preprocessing.text.Tokenizer(num_words=10000,char_level=True)
overview_token = tf.keras.preprocessing.text.Tokenizer(num_words=10000)


title_token.fit_on_texts(list2string(training_data[:,1]))
overview_token.fit_on_texts(list2string(training_data[:,0]))

seq_title = title_token.texts_to_sequences(training_data[:,1])
seq_overview = overview_token.texts_to_sequences(training_data[:,0])

print(seq_title[13])

print(title_token.texts_to_sequences(training_data[:,1]))

input_shape=(None, 1)
inputs = tf.keras.layers.Input(input_shape)
x=tf.keras.layers.Embedding(10000, 64, input_length=None)
x= tf.keras.layers.LSTM(3,activation="relu", return_sequences=True)(inputs)
x= tf.keras.layers.LSTM(4, activation="relu")(x)
output = tf.keras.layers.Dense(32, activation='relu')

overview_model = tf.keras.models.Model(inputs=inputs, output=output)

overview_model.summary()