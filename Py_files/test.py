import os
import cv2
import matplotlib.pyplot as plt
import utils
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np


model = keras.models.load_model('model.h5', custom_objects={'FocalLoss': utils.FocalLoss
                                                            , 'dice_coef': utils.dice_coef})

BASE_DIR = ''
TRAIN_DIR = BASE_DIR + 'cropped_img/'
MASK_DIR = BASE_DIR + 'cropped_mask/'

train = os.listdir(TRAIN_DIR)
mask = os.listdir(MASK_DIR)

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


X_train, X_test, y_train, y_test = train_test_split(
    out_rgb,
    out_mask,
    test_size=0.05,
    shuffle=True)


rows = 1
columns = 3
preds = model.predict(X_test)

for img, pred, g_truth in zip(X_test[10:15], preds[10:15], y_test[10:15]):
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction")
    fig.add_subplot(rows, columns, 3)
    plt.imshow(g_truth, interpolation=None)
    plt.axis('off')
    plt.title("Ground truth")