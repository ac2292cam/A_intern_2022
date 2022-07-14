import keras as keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

imageset = loadmat('exp_w109.mat')['DATAcropped']
print(f"imageset shape = {imageset.shape}")
imageset = np.transpose(imageset, (2,0,1))
print(f"imageset shape = {imageset.shape}")

# chops the images into pixel by pixel squares
# originalimages must be in format (number of images, x, y)
def chop(originalimages, xpixels, ypixels):
    xrange = np.arange(0,originalimages.shape[1]-xpixels,xpixels)
    print(xrange)
    yrange = np.arange(0,originalimages.shape[2]-ypixels,ypixels)
    print(yrange)
    choppedimages = np.zeros((len(xrange)*len(yrange)*originalimages.shape[0], xpixels, ypixels))

    index = 0
    for image in originalimages:
        for x in xrange:
            for y in yrange:
                choppedimages[index,:,:] = image[x:x+xpixels,y:y+ypixels]
                index += 1

    return choppedimages

#chop into xpixel by ypixel images
xpixels = 512
ypixels = 256
smallerimages = np.random.shuffle(chop(imageset, xpixels, ypixels))
print(f"smaller images shape = {smallerimages.shape}")
no_of_images = smallerimages.shape[0]
no_of_training = int(no_of_images*0.9)
no_of_test = int(no_of_images*0.1)

# print example of image
#plt.imshow(smallerimages[20,:,:])
#plt.imshow(smallerimages[100,:,:])
#plt.imshow(smallerimages[300,:,:])
#plt.imshow(smallerimages[500,:,:])

# rescale entries to [0,1] and add extra dimension
# use 90% for training and 10% for testing
X_train = smallerimages[0:no_of_training,:,:]/(2**16)
X_test = smallerimages[no_of_training:no_of_images,:,:]/(2**16)

imageshape = X_train.shape[1:3]

encoder = keras.models.Sequential([

    # Encoder
    keras.layers.Reshape([xpixels, ypixels, 1], input_shape = imageshape),

    keras.layers.Conv2D(filters =  16, kernel_size=8, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters =  32, kernel_size=8, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Conv2D(filters =  64, kernel_size=4, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(pool_size=2),
])

print(encoder.summary())

decoder = keras.models.Sequential([
    keras.layers.Conv2DTranspose(32, kernel_size = 3, strides =2, padding = 'same', activation ='relu', input_shape = [64,32,64]),
    keras.layers.Conv2DTranspose(16, kernel_size = 3, strides =2, padding = 'same', activation ='relu'),
    keras.layers.Conv2DTranspose(1, kernel_size = 3, strides =2, padding = 'same', activation ='tanh'),
    keras.layers.Reshape([xpixels,ypixels])
])

print(decoder.summary())

autoencoder = keras.models.Sequential([encoder,decoder])
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy', metrics = ['accuracy'])
autoencoder.fit(X_train, X_train, epochs = 5, batch_size = 32)

def plot_image(image):
    plt.imshow(image, cmap = "binary")
    plt.axis("off")

def show_reconstructions(model, n_images = 5):
    reconstructions = model.predict(X_test[:n_images])
    fig = plt.figure()
    for image_index in range(n_images):
        plt.subplot(2,n_images,1+image_index)
        plot_image(X_test[image_index])
        plt.subplot(2,n_images, 1+n_images +image_index)
        plot_image(reconstructions[image_index])
show_reconstructions(autoencoder)
plt.savefig("reconstructions")