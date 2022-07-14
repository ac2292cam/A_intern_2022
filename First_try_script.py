import tensorflow as tf
import tensorflow.keras.models.Sequential as Sequential
import tensorflow.keras.models.Model as Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Dropout
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



autoencoder = Sequential([

    # Encoder
    Conv2D(filters =  64, kernel_size=(7,7), activation='relu', padding='same', input_shape=imageshape, strides=(2,2))
    MaxPooling2D(pool_size=(2,2))

    Conv2D(filters = 128 , kernel_size=(3,3), activation='relu', padding='same')
    Conv2D(filters = 128 , kernel_size=(3,3), activation='relu', padding='same')
    MaxPooling2D(pool_size=(2,2), padding='same')

    Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding = 'same', strides=(2,2))
    Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding = 'same', strides=(2,2))
    MaxPooling2D(pool_size=(2,2), padding='same')
    Flatten()

    # Decoder

    Dense(128, activation='relu')
    Dropout(0.5))

    Dense(64, activation='relu')
    Dropout(0.5)
    Dense(10, activation='softmax')
    ])

print(autoencoder.summary())

print('finished setting up autoencoder')

autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
print('finished compiling')
autoencoder.fit(X_train, X_train, epochs = 5, batch_size=1)
print('finished training')

encoder = Model(inputs = autoencoder.input, outputs = autoencoder.get_layer('flatten_10').output)
coded_test_images = encoder.predict(X_test, batch_size=1)
decoded_test_images = autoencoder.predict(X_test, batch_size=1)

print('done')

#%%
