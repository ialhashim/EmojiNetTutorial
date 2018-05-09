#========================================================================================
# Hyperparameters
#========================================================================================
mynetname           = 'emojinet'
resolution          = 128
epochs              = 30
batch_size          = 64
lrate               = 0.0001
validation_split    = 0.2

eN                  = 32
dN                  = 64

# Specify GPUs
#import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

#========================================================================================
# Input
#========================================================================================
import dataset, numpy
x,y = dataset.load(resolution=resolution, count=10)

# Modify inputs
x = x / 255
y = y / 255

# Augment

print('x: ' + str(len(x)) + ' images of shape ' + str(x[0].shape))
print('y: ' + str(len(y)) + ' images of shape ' + str(y[0].shape))

#========================================================================================
# Outputs
#========================================================================================
import time, pathlib, utils

# Define output folder name
runID = str(int(time.time())) + '-n' + str(len(x)) + '-r' + str(resolution) + \
		'-eN' + str(eN) + '-dN' + str(dN) + '-e' + str(epochs) + \
		'-bs' + str(batch_size) + '-lr' + str(lrate) + '-' + mynetname

runPath = './Graphs/' + runID

print('Output: ' + runPath)
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True) 
pathlib.Path(runPath+'/checkpoints').mkdir(parents=True, exist_ok=True)
pathlib.Path(runPath+'/samples').mkdir(parents=True, exist_ok=True) 

#========================================================================================
# Neural network
#========================================================================================
import keras as keras
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Concatenate, LeakyReLU, MaxPooling2D, UpSampling2D
from keras import losses
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

# Expected input and output
in_shape = x.shape[1:]
out_shape = y.shape[1:]

# How to set the initial random weights of Keras layers
initializer = 'he_uniform'

# Define the input layer:
input_img = Input(shape=in_shape, name='input_img')

# Encoder
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV0')(input_img)
encoder = LeakyReLU(alpha=0.1)(encoder)
encoder = Conv2D(eN, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV1')(encoder)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL1   = MaxPooling2D((2, 2), name='POOL1') (encoder)
encoder = Conv2D(eN*2, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV2')(POOL1)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL2   = MaxPooling2D((2, 2), name='POOL2') (encoder)
encoder = Conv2D(eN*2, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV3')(POOL2)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL3   = MaxPooling2D((2, 2), name='POOL3') (encoder)
encoder = Conv2D(eN*4, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV4')(POOL3)
encoder = LeakyReLU(alpha=0.1)(encoder)
POOL4   = MaxPooling2D((2, 2), name='POOL4') (encoder)
encoder = Conv2D(eN*4, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV5')(POOL4)
encoder = LeakyReLU(alpha=0.1)(encoder)
encoder = MaxPooling2D((2, 2), name='POOL5') (encoder)
encoder = Conv2D(eN*8, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='ENC_CONV6')(encoder)
encoder = LeakyReLU(alpha=0.1)(encoder)

# Decoder
decoder = UpSampling2D((2, 2), name='UPSAMPLE5')(encoder)
decoder = keras.layers.concatenate([decoder,POOL4], name='CONCAT5')
decoder = Conv2D(dN*16, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV5A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN*16, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV5B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE4')(decoder)
decoder = keras.layers.concatenate([decoder,POOL3], name='CONCAT4')
decoder = Conv2D(dN*8, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV4A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN*8, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV4B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE3')(decoder)
decoder = keras.layers.concatenate([decoder,POOL2], name='CONCAT3')
decoder = Conv2D(dN*6, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV3A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN*6, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV3B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE2')(decoder)
decoder = keras.layers.concatenate([decoder,POOL1], name='CONCAT2')
decoder = Conv2D(dN*4, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV2A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(dN*4, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV2B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = UpSampling2D((2, 2), name='UPSAMPLE1')(decoder)
decoder = keras.layers.concatenate([decoder,input_img], name='CONCAT1')
decoder = Conv2D(64, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV1A')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(32, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer, name='DEC_CONV1B')(decoder)
decoder = LeakyReLU(alpha=0.1)(decoder)
decoder = Conv2D(1, (1, 1), activation='sigmoid')(decoder)

# Single GPU
model = Model(inputs=input_img, outputs=decoder, name=runID)
model.summary()
#plot_model(model, to_file='./Graphs/'+runID+'.png', show_shapes=True, show_layer_names=True)

# Optimizer:
optimizer = keras.optimizers.Adam(lr=lrate)

# Compile model with specified loss function:
model.compile(loss=utils.bce_dice_loss, optimizer=optimizer)

# Tensorboard:
tbCallBack = keras.callbacks.TensorBoard(log_dir=runPath, histogram_freq=0, write_graph=True, write_images=True)

# Checkpoints:
checkpointCallBack = keras.callbacks.ModelCheckpoint(filepath=runPath+'/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', save_weights_only=True, verbose=1, period=50)

# Test prediction result during training:
def testPrediction(epoch):
	if epoch % 5 == 0:
		for i in range(0,len(x),int(len(y)/5)):
			if epoch == 0:
				utils.save_png(x[i]*255,runPath+'/samples/i'+f'{i:04}'+'.0.rgb.png')
				utils.save_png(y[i]*255,runPath+'/samples/i'+f'{i:04}'+'.1.mask.png')
			prediction = utils.predict(model, x[i])*255
			utils.save_png(prediction.reshape(resolution,resolution,3), runPath+'/samples/i'+f'{i:04}'+'.e'+f'{epoch:05}'+'.png')
testPredictCallBack = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: testPrediction(epoch))

# Start training:
history = model.fit(x, y, callbacks=[tbCallBack,checkpointCallBack,testPredictCallBack], epochs=epochs, batch_size=batch_size, validation_split=validation_split)

# Save trained model:
model.save(runPath+'/model.h5')