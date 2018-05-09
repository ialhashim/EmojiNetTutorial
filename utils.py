import numpy, cv2

# Predict for a single image
def predict(model, image):
    prediction = model.predict(image.reshape(1, image.shape[0], image.shape[1], image.shape[2]))
    return numpy.tile(prediction,3)

# Save image to PNG
def save_png(image, filename):
    if image.shape[2] == 3:
        image = cv2.cvtColor(image.astype(numpy.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, numpy.clip(image,0,255))

# Loss function for UNet
from keras.losses import binary_crossentropy
import keras.backend as K

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss