---
title: Balance Classes During YOLO Training Using a Weighted Dataloader
seo_title: Balance Classes During YOLO Training Using a Weighted Dataloader
summary: Implement class balancing in Ultralytics using a weighted dataloader and improve the performance of minority class.
description: Implement class balancing in Ultralytics using a weighted dataloader and improve the performance of minority class without duplicating or removing any data.
slug: yolo-class-balancing
author: Mohammed Yasin

draft: false
date: 2024-09-01T22:15:51+08:00
lastmod: 
expiryDate: 
publishDate: 

feature_image: class_balancing.png
feature_image_alt: class balancing with weighted dataloader

categories:
  - Tutorials
tags:
  - YOLO
  - Object Detection
series:

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

## Introduction

Oftentimes, the dataset you have is not balanced. You might have some classes
that are underrepresented and some that are overrepresented. It sometimes helps
to account for these imbalances during the training process. One of the ways to
accomplish this is to use a weighted dataloader. Compared to other approaches,
the weighted dataloader has some key benefits:
1. **Simplicity:** Unlike adding loss weights, which can get tricky and are tied to
   the specific loss function, a weighted dataloader is straightforward and
   easier to implement, as you’ll see.
2. **Preserve All Data:** You don’t have to undersample the majority class to
   balance things out. You get to use all your data, which is a big plus.
3. **Smoother Weight Updates:** With loss weights, the model might not
   frequently see batches containing minority classes, leading to sudden loss
   spikes when it does. A weighted dataloader avoids these sharp gradient
   shifts by ensuring a more consistent representation of all classes
   throughout training.

In this guide, we'll walk through implementing a weighted dataloader for YOLO
in Ultralytics, but the approach should work with other Ultralytics models that
use the same data loading class. You can find the full Colab notebook
[here](https://colab.research.google.com/drive/1hnQmrDGKBIF8Vm_C73fsswQK7QMkUw2Q?usp=sharing).

## Implementation

To implement the dataloader, we start by creating a new `Dataset` class that
inherits from the existing `YOLODataset` class. In this new class, we add the
necessary methods to handle weight balancing and override the `__getitem__`
method to return images based on the calculated probabilities for each one.
This ensures that the dataloader provides a more balanced distribution
of images during training.

```python
class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()
    
    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
              weights.append(1)
              continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))
```

The `Detect`, `Segment` and `Pose` models are usually trained using labels that
have multiple objects. So we have to somehow aggregate the weights of all the
labels in an image to get the final weight for that image. In the above
implementation, we use `np.mean`, but you can try other aggregation functions
such as `np.max`, `np.median`, `np.sum` or even something custom and more
nuanced.

We can then just monkey-patch the `YOLODataset` class in `build.py` like so:

```python
import ultralytics.data.build as build

build.YOLODataset = YOLOWeightedDataset
```

While there are more Pythonic ways to make the training use the custom class, I
found this method to be the simplest and most versatile. It works across all
YOLO models in Ultralytics, including `Detect`, `Segment`, and `Pose`, and even
extends to `RTDETR`.

## Testing the weighted dataloader

I ran a short test on a [traffic lights dataset](https://universe.roboflow.com/valorem-reply-diswd/traffic-lights-set-1/dataset/1) that I found on Roboflow, which has the following class distribution:

<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center;">
    <img src="/tutorials/yolo-class-balancing/label_counts.jpg" alt="class distribution of the traffic lights dataset" style="max-width: 400px; height: auto;">
    <p><b>Class distribution of the traffic lights dataset</b></p>
  </div>
</div>

As you can see, the `Yellow Light` and `Left Turn` classes are
underrepresented, making this an ideal scenario to test the weighted dataloader
we created.

In the first training run, we didn’t use the weighted dataloader. We trained
for 50 epochs using `yolov8n.pt` as the starting checkpoint, resulting in the
following metrics:

```bash
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:06<00:00,  1.01it/s]
                   all        199        495      0.764      0.604      0.647      0.357
           Green Light        123        260      0.822      0.693      0.755      0.397
             Left turn         36         65       0.91      0.619      0.688      0.387
             Red Light         61        135      0.728      0.595      0.589      0.301
          Yellow Light         15         35      0.598       0.51      0.556      0.343
```

For the next training run, we switched to the weighted dataloader and similarly
trained it for 50 epochs. The results were the following:

```bash
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 7/7 [00:06<00:00,  1.07it/s]
                   all        199        495      0.831      0.601      0.673      0.351
           Green Light        123        260      0.895      0.658      0.754      0.383
             Left turn         36         65      0.906      0.596      0.692      0.351
             Red Light         61        135      0.811      0.578      0.623       0.32
          Yellow Light         15         35      0.712      0.571      0.622      0.351
```

The overall mAP50 increased by 2.5, with the most significant improvement in
the `Yellow Light` class, one of the underrepresented classes, where the mAP50
jumped by 6.6.

Additionally, you’ll notice that more images from the minority classes appear
in the training batch plots when using the weighted dataloader, compared to the
default dataloader:

<div style="display: flex; justify-content: center; gap: 10px;">
  <div style="flex: 1; text-align: center;">
    <img src="/tutorials/yolo-class-balancing/batch_unweighted.jpg" alt="training batch from default dataloader" style="max-width: 100%; height: auto;">
    <p><b>Default Dataloader</b></p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/tutorials/yolo-class-balancing/batch_weighted.jpg" alt="training batch from weighted dataloader" style="max-width: 100%; height: auto;">
    <p><b>Weighted Dataloader</b></p>
  </div>
</div>

## Is it perfectly balanced?

Since an image can have multiple labels, you can't always individually
increase the instances of one particular class to make them perfectly balanced,
because if an image has both a majority class and a miniority class, using
that image in the batch will increase the instances of both the majority and
minority class in the batch.

From the testing, I noticed that using `np.sum` aggregation resulted in a
somewhat more balanced class distribution than `np.mean`:

<div style="display: flex; justify-content: center; gap: 10px;">
  <div style="flex: 1; text-align: center;">
    <img src="/tutorials/yolo-class-balancing/weighted_mean_dist.png" alt="class distribution when using mean" style="max-width: 100%; height: auto;">
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="/tutorials/yolo-class-balancing/weighted_sum_dist.png" alt="class distribution when using sum" style="max-width: 100%; height: auto;">
  </div>
</div>

But this can vary from one dataset to another, so it would be good test
different aggregation functions to see what works best for your dataset.

## Conclusion

In this guide, we demonstrated how easy it is to create and implement a
weighted dataloader and seamlessly integrate it with the `ultralytics` library.
We avoided any complicated changes to loss functions and didn’t even need to
modify the source code directly. However, it's worth noting that using loss
function weights has an advantage in scenarios where images contain multiple
classes, as it can be challenging to perfectly balance those with a weighted
dataloader. Experimenting with different aggregation functions might help you
achieve the best results for your dataset.

Thanks for reading.
