
import numpy as np

import keras.backend as K


def make_image_gen(X_train, y_train, aug, batch_size):
    aug_x = []
    aug_y = []
    while True:
        for i in range(X_train.shape[0]):
            augmented = aug(image=X_train[i], mask=y_train[i])
            x, y = augmented['image'],  augmented['mask']
            aug_x.append(x)
            aug_y.append(y)
            if len(aug_x)>=batch_size:
                yield np.array(aug_x, dtype = 'float32'), np.array(aug_y, dtype = 'float32')
                aug_x, aug_y=[], []

ALPHA = 0.8
GAMMA = 2

def FocalLoss(targets, inputs, alpha=ALPHA, gamma=GAMMA):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    BCE = K.binary_crossentropy(targets, inputs)
    BCE_EXP = K.exp(-BCE)
    focal_loss = K.mean(alpha * K.pow((1 - BCE_EXP), gamma) * BCE)

    return focal_loss

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice