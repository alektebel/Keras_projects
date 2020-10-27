import tensorflow as tf

input_shape=(None, 32, 32, 3)
class NiceLayer(tf.keras.layers.Layer):

    def __init__(self, input_dim, units):
        super(NiceLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(input_dim, units))

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


class NiceConvo(tf.keras.models.Model):

    def __init__(self, num_classes, **kwargs):
        super(NiceConvo, self).__init__(**kwargs)
        self.nicelayer = NiceLayer(10, 4)
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2= tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        h = self.dense1(inputs)
        h=self.dropout1(h)
        h=self.nicelayer(h)
        return self.dense2(h)


my_model = NiceConvo(10)
my_model.compile(optimizer='Adam',
                 loss='mse')
my_model.build(input_shape)
my_model.summary()



