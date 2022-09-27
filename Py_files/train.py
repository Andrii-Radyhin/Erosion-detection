import os

import cv2
import numpy as np

import tensorflow as tf
import keras.backend as K
from tensorflow import keras
import utils
import albumentations as A
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()

BASE_DIR = ''
TRAIN_DIR = BASE_DIR + 'cropped_img/'
MASK_DIR = BASE_DIR + 'cropped_mask/'
INPUT_SIZE = (256,256,3)

train = os.listdir(TRAIN_DIR)
mask = os.listdir(MASK_DIR)

print(f"Train files: {len(train)}. ---> {train[:3]}")
print(f"Test files :  {len(mask)}. ---> {mask[:3]}")

out_rgb = []
out_mask = []

counter_empty = 0

for p_img, p_mask in zip(train, mask):
    img_path = os.path.join(TRAIN_DIR, p_img)
    mask_path = os.path.join(MASK_DIR, p_mask)

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.
    mask = cv2.imread(mask_path)

    mask = mask[:, :, :1]
    mask[mask > 0.] = 1.

    if 1 not in mask: counter_empty += 1

    if not (1 not in mask and counter_empty >= 40):
        out_rgb += [img]
        out_mask += [mask]

out_rgb = np.array(out_rgb, dtype='float32')
out_mask = np.array(out_mask, dtype='float32')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    out_rgb,
    out_mask,
    test_size=0.05,
    shuffle=True)

aug = A.Compose([
    A.OneOf([
        A.RandomSizedCrop(min_max_height=(50, 101), height=256, width=256, p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, p=0.5)
    ],p=1),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.OneOf([
        A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.GridDistortion(p=0.5),
        A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
    ], p=0.8)])


ALPHA = 0.8
GAMMA = 2


callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5),
    tf.keras.callbacks.ModelCheckpoint(filepath='crusa.h5', monitor = 'val_loss', verbose = 1,
                                       save_best_only = True, mode = 'min'),

    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-6)
]

model = sm.Unet('efficientnetb0', classes=1, input_shape=(256, 256, 3),
                activation='sigmoid', encoder_weights='imagenet')

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=utils.FocalLoss, metrics = [utils.dice_coef] )
model.fit_generator(generator=utils.make_image_gen(X_train, y_train, aug, 16),epochs=50,
                    steps_per_epoch = 200, callbacks = callbacks,validation_data = (X_test, y_test))

model.save('model.h5')
print('Model saved as model.h5, use test.py to load it')