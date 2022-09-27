# Satellite-Segmentation
Abstract: this repo includes a pipeline using tf.keras for training Unet + EfficientNetb0 for the problem of erosion detection. Moreover, weights and trained model are provided. I will use EDA.ipynb as main file in EDA below, all images from output EDA.ipynb.

**Data**: [tap here](https://drive.google.com/drive/folders/1_T-R-FvMaNDeawhHGtUZ6Dc8KF4ERNrn?usp=sharing)

## EDA
Important to notice that we have dataset in .jp2 format (Satellite image), EDA.ipynb contains code using [rasterio](https://rasterio.readthedocs.io/en/latest/)
based on [article](https://medium.datadriveninvestor.com/preparing-aerial-imagery-for-crop-classification-ce05d3601c68).

Aftеr article code execution in EDA.ipynb we got 2 images with shape (3, 10980, 10980) for image and (1, 10980, 10980) for mask.
My suggestion is to simply crop it into images with shape (256, 256) but there is also option for shape (512, 512).

Next, we must predict that cropped mask might contain empty masks, EDA confirms it:
```sh
All 256: 1764. ---> Empty: 1500
All 512:  441. ---> Empty 306
```
As there is not really balanced dataset, let's extract images with non-empty masks using either randomizer or just first 40 images. 40 images because it's 20% of images with non-empty masks. So we have 264 + 40 = 304 (for (256,256) crop) images.

**But there is still one question not answered: what crop is better?**

By this reason we will use pixel ratio (mask pixels/ all mask pixels).
Also graph below shows pixel ratio with all images, received during cropping.

![alt text](images/ratio_shape.PNG)

Seems like it's better (512,512) cropping, from another hand it's only 141 non-empty masks, therefore was decided to use (256,256) one.

**Influence on model predictions:** at the end of EDA should to say: during training, i noticed that masks actually not cover all erosion:

![alt text](images/Masks_problem.PNG)

Now we have 304 images for trainig, it's quite small number of images, as a solution i suggest to use albumentations to create artificial images, it will be done in training file.

 

