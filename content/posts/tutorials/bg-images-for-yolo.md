---
title: How to download background images for YOLO model training?
seo_title: Downloading Background Images For YOLO Model Training
summary: Guide to download background images using COCOAPI.
description: A simple guide to downloading background images to reduce false positives in object detection models.
slug: bg-images-for-yolo
author: Mohammed Yasin

draft: false
date: 2023-08-06T09:54:00+08:00
lastmod:
expiryDate: 
publishDate: 

feature_image: 
feature_image_alt: 

categories:
  - Tutorials
tags:
  - COCOAPI
  - YOLO
  - Object Detection

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

## Introduction

One prominent problem in object detection is the presence of false positives. False positives are objects that are detected by the model but are not actually present in the image.

A way of reducing such false positives is to train the model on negative images. Negative images are images that do not contain any objects. This helps the model learn what an image without objects looks like, so that it understands what is to be considered an object and what is to be regarded as background. In fact, the YOLOv8 guide suggests that the model be trained on 10% negative images.

In the following sections, I explain how to download negative images using the COCOAPI.

## Prerequisites

The COCOAPI is a Python API that allows you to easily explore the COCO dataset by providing useful methods to select and filter images based on the object class and other metadata that is associated with each object in an image. Consequently, we can leverage it to specifically filter and download images that would act as negative images for our use-case.

To install the COCOAPI, run the following command in your terminal:
  
  ```bash
  pip install pycocotools
  ```

Besides the COCOAPI, you also need to download the `instances_train2017.json` COCO annotations file which would be loaded into the API to perform the filtering necessary. You can download the `.zip` file containing the COCO annotations from [here](https://cocodataset.org/#download) under the **Annotations** section labeled **2017 Train/Val annotations**. Extract the downloaded file to get the `instances_train2017.json` file.

## Downloading the images

1. Fire up your Jupyter Notebook or the Python console.
2. Import the necessary packages:

    ```python
    from pycocotools.coco import COCO
    import requests
    import os
    ```

3. Define the classes to be excluded from the downloaded images. This is because we can't use images that contain classes that we are trying to detect as background images as that would cause false negatives.

    The COCO classes are listed [here](https://github.com/ultralytics/yolov5/blob/df48c205c5fc7be5af6b067da1f7cb3efb770d88/data/coco.yaml). You will have to find the classes that you are detecting from this list and use that exact same label in the `exclude_classes` list. For example, if you are detecting cars, you will have to use the label `car` and not `cars` or `Car` or `Cars`. This includes classes that may be similar to the classes present in COCO. For example, if you are detecting `pedestrians`, you will have to exclude the label `person` as it is similar to `pedestrian`.

    ```python
    DETECTOR_CLASSES = ['person', 'car'] # Specify the COCO classes that you are detecting
    ```

4. Next, define the number of background images to be downloaded.

    ```python
    NUM_IMAGES = 1000 # Number of background images to download
    ```

5. Finally, run the following script which would download and save the images and the corresponding blank label files to `images` and `labels` folders respectively.

    ```python
    # Adapted from https://stackoverflow.com/a/62770484/8061030

    # instantiate COCO specifying the annotations json path
    coco = COCO("instances_train2017.json")

    # Specify a list of classes to exclude.
    # Background images will not contain these.
    # These should be classes included in training.
    exc_cat_ids = coco.getCatIds(catNms=DETECTOR_CLASSES)

    # Get the corresponding image ids and images using loadImgs
    exc_img_ids = coco.getImgIds(catIds=exc_cat_ids)

    # Get all image ids
    all_cat_ids = coco.getCatIds(catNms=[""])
    all_img_ids = coco.getImgIds(catIds=all_cat_ids)

    # Remove img ids of classes that are included in training
    bg_img_ids = set(all_img_ids) - set(exc_img_ids)

    # Get background image metadata
    bg_images = coco.loadImgs(bg_img_ids)

    # Create dirs
    os.makedirs("images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)

    # Save the images into a local folder
    for im in bg_images[:NUM_IMAGES]:
        img_data = requests.get(im["coco_url"]).content

        # Save the image
        with open("images/" + im["file_name"], "wb") as handler:
            handler.write(img_data)

        # Save the corresponding blank label txt file
        with open("labels/" + im["file_name"][:-3] + "txt", "wb") as handler:
            pass
    ```

## Conclusion

That's it. You should have the necessary background images downloaded now. You can use these images to train your object detection model. The downloaded images and labels are in YOLO (Darknet) format. If you want to convert the images and labels to a different format, you can try [datumaro](https://openvinotoolkit.github.io/datumaro/latest/docs/command-reference/context_free/convert.html#convert). For example, to convert it to the YOLO Ultralytics format, you would run:

```bash
datum convert -i bg_dataset -if yolo -f yolo_ultralytics -o yolo_v8_dataset -- --save-media
```

Alternatively, you could simply manually copy and paste the images and labels into the images/train and labels/train folder respectively of an already existing YOLOv8 dataset.

Thanks for reading!
