import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard
import numpy as np 


# this is the size of our encoded representations
encoding_dim = 3

x = np.array([[1,2,3],
              [1,2,3],
              [1,2,3],
              [1,2,3],
              [1,2,3],
              [1,2,3],
              [1,2,3],
              [1,2,3],
              [1,2,3]])

inputs = Input(shape=(3,))
encoded = Dense(encoding_dim, activation='relu')(inputs)
decoded = Dense(3)(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adadelta', loss='mse')

log_dir = './tmp'
tbCallBack = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

autoencoder.fit(x, x,
                epochs=500,
                batch_size=4,
                callbacks=[tbCallBack])
