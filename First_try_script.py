import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten
from scipy.io import loadmat

first100images = loadmat('w135_first100images.mat')['DATAcropped']
print(first100images.shape)
first100images = np.transpose(first100images, (2,0,1))
print(first100images.shape)

X_train_original = first100images[0:90,:,:].reshape((90, 3320, 399, 1))/(2**16)
X_test_original = first100images[90:100,:,:].reshape((10, 3320, 399, 1))/(2**16)
X_train = np.zeros((90, 3320, 400, 1))
X_test = np.zeros((10, 3320, 400, 1))
X_train[:,:,:399,:] = X_train_original
X_test[:,:,:399,:] = X_test_original

imageshape = X_train.shape[1:4]

autoencoder = Sequential()

# Encoder
autoencoder.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu', padding='same', input_shape=imageshape))
autoencoder.add(MaxPooling2D(pool_size=(2,2)))

autoencoder.add(Conv2D(filters =16 , kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(MaxPooling2D(pool_size=(2,2), padding='same'))

autoencoder.add(Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding = 'same', strides=(2,2)))
autoencoder.add(Flatten())

# Decoder

autoencoder.add(Reshape((415,50,8)))

autoencoder.add(Conv2D(filters = 8, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))

autoencoder.add(Conv2D(filters = 16, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))

autoencoder.add(Conv2D(filters = 32, kernel_size=(3,3), activation='relu', padding='same'))
autoencoder.add(UpSampling2D(size=(2,2)))

autoencoder.add(Conv2D(filters = 1, kernel_size=(3,3), activation='sigmoid', padding='same'))
print(autoencoder.summary())

print('done set up autoencoder')

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
print('done compile')
autoencoder.fit(X_train, X_train, epochs = 5)
print('finished training')

encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('flatten_10').output)
coded_test_images = encoder.predict(X_test)
decoded_test_images = autoencoder.predict(X_test)

print('finish')