#========================================================================================
# Hyperparameters
#========================================================================================
mynetname           = "emojinet"
resolution_RGB      = 128
resolution_Depth    = 128
epochs              = 100
batch_size          = 64
lrate               = 0.0001
initializer         = "he_uniform"
validation_split    = 0.2

eN = 96
dN = 192

#========================================================================================
# Inputs
#========================================================================================
import numpy as np
import os
import cv2
import h5py
from urllib import request

x,y = [],[]

#print(os.listdir("."))

for di in range(10):
  
  dataset_filename = 'dataset'+f'{di:03}'+'.hdf5'
  print("Loading: " + dataset_filename)
  
  # Download datasets if needed
  url_root = 'https://s3-eu-west-1.amazonaws.com/deepemoji/'
  if not os.path.isfile(dataset_filename):
    url = url_root + dataset_filename
    f = open(dataset_filename, 'wb')
    f.write(request.urlopen(url).read())
    f.close()
    print("Downloaded ["+url+"]")
  
  # Open dataset and collect all data
  with h5py.File(dataset_filename, "r") as dataset:
    keys = []
    for key in dataset.keys(): keys.append(key.split(".")[0])
    keys = list(set(keys))

    for key in keys:
      rgb, mask = dataset.get(key+'.rgb').value, dataset.get(key+'.mask').value

      rgb = cv2.resize(rgb, dsize=(resolution_RGB, resolution_RGB))
      mask = cv2.resize(mask, dsize=(resolution_RGB, resolution_RGB))

      
      x.append(rgb)
      y.append(mask.reshape((mask.shape[0],mask.shape[1],1)))

x = np.stack(x) 
y = np.stack(y) 

# Augment
#GRB = x[...,[1,0,2]].copy()
#GRB_mask = y.copy()
#x = np.concatenate((x, GRB))   
#y = np.concatenate((y, GRB_mask))

# Modify inputs
x = x
y = y

print('x: ' + str(len(x)) + ' images of shape ' + str(x[0].shape))
print('y: ' + str(len(y)) + ' images of shape ' + str(y[0].shape))

in_shape = x.shape[1:]
out_shape = y.shape[1:]

print(in_shape)
print(out_shape)

#========================================================================================
# Outputs
#========================================================================================
import pathlib
import time

runID = str(int(time.time())) + "-n" + str(len(x)) + "-r" + str(resolution_RGB)+"-d"+str(resolution_Depth) + "-eN" + str(eN) + "-dN" + str(dN) + "-e" + str(epochs) + "-bs" + str(batch_size) + "-lr" + str(lrate) + "-" + mynetname

runPath = "./Graphs/" + runID

print("Output: " + runPath)
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True) 
pathlib.Path(runPath+"/checkpoints").mkdir(parents=True, exist_ok=True)

import keras as keras
from keras import losses
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Conv2D, Concatenate, LeakyReLU, MaxPooling2D, UpSampling2D
from keras.utils.vis_utils import plot_model


# Optimizer:
optimizer = keras.optimizers.Adam(lr=lrate)

# Tensorboard:
tbCallBack = keras.callbacks.TensorBoard(log_dir=runPath, histogram_freq=0, write_graph=True, write_images=True)

# Checkpoints:
checkpointCallBack = keras.callbacks.ModelCheckpoint(filepath=runPath+'/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_weights_only=True, verbose=1, period=50)

# Find input image shape
input_img = Input(shape=in_shape, name='input_img')

# Encoder
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV0')(input_img)
encoder = LeakyReLU(alpha=0.1)(encoder)
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV1')(encoder)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL1   = MaxPooling2D((2, 2), name='POOL1') (encoder)
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV2')(POOL1)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL2   = MaxPooling2D((2, 2), name='POOL2') (encoder)
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV3')(POOL2)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL3   = MaxPooling2D((2, 2), name='POOL3') (encoder)
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV4')(POOL3)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL4   = MaxPooling2D((2, 2), name='POOL4') (encoder)
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV5')(POOL4)
encoder = LeakyReLU(alpha=0.1)(encoder)
encoder = MaxPooling2D((2, 2), name='POOL5') (encoder)
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV6')(encoder)
encoder = LeakyReLU(alpha=0.1)(encoder)

# Decoder
decoder = UpSampling2D((2, 2), name='UPSAMPLE5')(encoder)
decoder = keras.layers.concatenate([decoder,POOL4], name='CONCAT5')
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV5A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV5B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE4')(decoder)
decoder = keras.layers.concatenate([decoder,POOL3], name='CONCAT4')
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV4A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV4B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE3')(decoder)
decoder = keras.layers.concatenate([decoder,POOL2], name='CONCAT3')
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV3A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV3B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE2')(decoder)
decoder = keras.layers.concatenate([decoder,POOL1], name='CONCAT2')
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV2A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV2B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE1')(decoder)
decoder = keras.layers.concatenate([decoder,input_img], name='CONCAT1')
decoder = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV1A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV1B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(out_shape[2], kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV1C')(decoder)

# Single GPU
model = Model(input_img, decoder, name=runID)
model.summary()
#plot_model(model, to_file='./Graphs/'+runID+'.png', show_shapes=True, show_layer_names=True)

# Compile model with specified loss
model.compile(loss=losses.mean_squared_error, optimizer=optimizer)

# Start training:
history = model.fit(x, y, callbacks=[tbCallBack,checkpointCallBack], epochs=epochs, batch_size=batch_size, validation_split=validation_split)

# Save trained model
model.save(runPath+'/model.h5')
