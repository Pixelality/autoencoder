
import numpy as np
import os
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from PIL import Image
import math
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard


IMAGE_FOLDER = 'C:/Users/Pixelality/Desktop/Autocompiler/catdonkey/'
LAYER_COUNT = 7
COMPILED_SIZE = 1
#LEARNING_RATE = 0.2
EPOCHS = 1000
FILTER = (3, 3)
POOLING = (2, 2)
image_shape = ()
image_size = 0
img_height = 0
img_width = 0


images = []

# load images into an (1, image size) shaped numpy array
for filename in os.listdir(IMAGE_FOLDER):
    img = Image.open(IMAGE_FOLDER+filename)
    if len(np.array(img.getdata()).shape) > 1:
        np_img = np.multiply(1 / 255, (np.array(img.getdata())[:, 0]).astype('float32'))
    else:
        np_img = np.multiply(1/255, (np.array(img.getdata())).astype('float32'))
    np_img = np.reshape(np_img, (img.height, img.width, 1))
    img_height = img.height
    img_width = img.width
    image_shape = np_img.shape
    #np_img = np_img.flatten()
    #image_size = len(np_img.flatten())
    images.append(np_img)
    #images.append(np.reshape(np_img, (1, len(np_img)))[0])
    image_size = int((img.height + img.width)/2)

with tf.device('/gpu:0'):
    input_img = Input(shape=image_shape)
    dim_list = []
    poollist = []
    for i in range(LAYER_COUNT-1):
        poollist.append(False)
        if i == 0:
            dim_list.append(image_size-int((image_size-COMPILED_SIZE)/LAYER_COUNT))
            x = Conv2D(dim_list[-1], FILTER, activation='relu', padding='same')(input_img)
            if img_height % 2**(i+1) == 0 and img_width % 2**(i+1) == 0 or img_width == img_height:
                x = MaxPooling2D(POOLING, padding='same')(x)
                poollist[-1] = True
        else:
            dim_list.append(dim_list[-1] - int((dim_list[-1]-COMPILED_SIZE)/(LAYER_COUNT-i)))
            x = Conv2D(dim_list[-1], FILTER, activation='relu', padding='same')(x)
            if img_height % 2**(i+1) == 0 and img_width % 2**(i+1) == 0 or img_width == img_height:
                x = MaxPooling2D(POOLING, padding='same')(x)
                poollist[-1] = True
    poollist.append(False)
    x = Conv2D(COMPILED_SIZE, FILTER, activation='relu', padding='same', name='middle')(x)
    if img_height % 2**(LAYER_COUNT) == 0 and img_width % 2**(LAYER_COUNT) == 0 or img_width == img_height:
        x = MaxPooling2D(POOLING, padding='same')(x)
        poollist[-1] = True

    encoder = x
    print(dim_list)
    for i in range(LAYER_COUNT-1):
        x = Conv2D(dim_list[-1-i], FILTER, activation='relu', padding='same')(x)
        if poollist[-1-i]:
            x = UpSampling2D(POOLING)(x)
    x = Conv2D(image_size, FILTER, activation='relu', padding='same')(x)
    if poollist[0]:
        x = UpSampling2D(POOLING)(x)
    decoder = Conv2D(1, FILTER, activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoder)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.fit(np.array(images), np.array(images),
                    epochs=EPOCHS,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(np.array(images), np.array(images)),
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    decoded_imgs = autoencoder.predict(np.array(images))
n = len(images)
plt.figure(figsize=(20, 4))
for i in range(0, n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(images[i].reshape(img_height, img_width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, (i+1) + n)
    plt.imshow(decoded_imgs[i].reshape(img_height, img_width))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

middle = 0
i = 0
while True:
    if autoencoder.get_layer(index=-1-i).output_shape[1] == img_height/2**LAYER_COUNT:
        middle = i
        break
    i += 1

inp = autoencoder.get_layer(index=-1-middle).input_shape[1:]
inp = Input(shape=(inp[0], inp[1], 1))
x = autoencoder.get_layer(index=-1-middle)(inp)
for i in range(1, middle+1):
    x = autoencoder.get_layer(index=-1-(middle-i))(x)

decode = Model(inp, x)

def update(val):
    vals = []
    """"for s in slider:
        vals.append(s.val)"""
    for i in range(decode.get_input_shape_at(0)[1]):
        vals.append([])
        for i2 in range(decode.get_input_shape_at(0)[2]):
            vals[-1].append([slider[i*decode.get_input_shape_at(0)[2]+i2].val])
    plt.subplot(211)
    image = decode.predict(np.array([vals]))
    plt.imshow(image.reshape(img_height, img_width))


axcolor = 'lightgoldenrodyellow'
slider = []
"""for i in range(int(img_height/2**LAYER_COUNT)):"""
for i in range(decode.get_input_shape_at(0)[2]*decode.get_input_shape_at(0)[1]):
    ax = plt.axes([0.25, (decode.get_input_shape_at(0)[2]*decode.get_input_shape_at(0)[1]-i)*0.1, 0.65, 0.03], facecolor=axcolor)
    slider.append(plt.Slider(ax, 'input'+str(i), 0, 30))
    slider[-1].on_changed(update)

update(0)
plt.show()