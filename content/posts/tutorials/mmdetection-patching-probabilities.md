---
title: Monkey Patching MMDetection 3.1.0 To Get Class Probabilities
seo_title: Monkey Patching MMDetection 3.1.0 To Get Class Probabilities
summary: A quick monkey patching technique to modify MMDetection 3.1.0 to return class probabilities.
description: A quick monkey patching technique to modify MMDetection 3.1.0 to return class probabilities.
slug: mmdetection-patching-probabilities
author: Mohammed Yasin

draft: false
date: 2023-08-17T01:01:00+08:00
lastmod:
expiryDate:
publishDate:

feature_image:
feature_image_alt:

categories:
  - Tutorials
tags:
  - MMDetection
  - Python
  - Object Detection

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

## Introduction

There are numerous frameworks that have been built to unify and streamline training and evaluation of models. MMDetection is one such framework for training detection models, and it is often praised for its modularity and massive collection of pre-trained models. Interestingly, it has also become a favorite among researchers with many authors publishing their codes based on MMDetection. This could also be attributed to how it makes the whole process of defining the dataset, the model architecture and the training and testing configuration as easy as attaching lego bricks.

Consequently, I decided that it would be best to use MMDetection for my research too, since it would make my life easier when I have to compare with other works which are mostly based on MMDetection. Anyway, turns out MMDetection, although good and flexible at what it does, is also very inflexible when it comes to making it do things that it doesn't do. I wanted to get all the classification probabilities associated with each bounding box prediction, I found that there was not a native way to do so. The framework also recently received an overhaul, deprecating many of the online tricks that people came up with to make this work. So I had to hack up my own method.

## Where art thou, `model.predict()`?

I started my goose chase from the `inference_detector()` function, which is a function provided by MMDetection that seamlessly takes an image path and returns you the prediction results in a nicely packed `DetDataSample` structure with all the inference results post-processed, and of course to my disappointment, that also means it throws away the class probabilities, and only returns the best probability associated with each predicted bounding box. After a wild chase of function calls, looking for the place where these probabilities get removed, I finally found the code lying bare in the `mmdet/models/detectors/single_stage.py` file. The file would however vary depending on the type of detector you're using. In my case it was RetinaNet which inherits from the `SingleStageDetector` class defined in the file that was just mentioned. Looking at the files that are defined as `_base` in your model config file should give you a clue.

## Monkey Patching

If you haven't heard of monkey patching, then you're in for a treat. It is one of my favorite features of Python. Monkey patching allows you to modify any Python function or method at runtime. This means I can replace a method of an instance with my own implementation of the said method, aka. monkey patch it without touching the source files at all.

This is useful because you don't have to worry about rewriting huge chunks of code or crudely editing the source files to get your modifications working. You can simply replace the function you want to modify with the modified version through monkey patching and it would behave almost as if it was always part of the code.

In this case, we're looking to monkey patch the `_predict_by_feat_single()` method where the class probabilities get filtered and thrown out. We have to modify it so that it doesn't get filtered. And the modifications are actually pretty straightforward:

```diff
@@ -61,6 +61,7 @@ def _predict_by_feat_single(self,
     mlvl_bbox_preds = []
     mlvl_valid_priors = []
     mlvl_scores = []
+    mlvl_raw_scores = []   # MODIFIED: create list to store raw scores 
     mlvl_labels = []
     if with_score_factors:
         mlvl_score_factors = []
@@ -94,6 +95,9 @@ def _predict_by_feat_single(self,
         # `nms_pre` than before.
         score_thr = cfg.get('score_thr', 0)
 
+        # MODIFIED: copy raw scores
+        raw_scores = scores
+
         results = filter_scores_and_topk(
             scores, score_thr, nms_pre,
             dict(bbox_pred=bbox_pred, priors=priors))
@@ -105,6 +109,9 @@ def _predict_by_feat_single(self,
         if with_score_factors:
             score_factor = score_factor[keep_idxs]
 
+        # MODIFIED: store raw scores
+        raw_scores = raw_scores[keep_idxs]
+
         mlvl_bbox_preds.append(bbox_pred)
         mlvl_valid_priors.append(priors)
         mlvl_scores.append(scores)
@@ -121,6 +128,7 @@ def _predict_by_feat_single(self,
     results = InstanceData()
     results.bboxes = bboxes
     results.scores = torch.cat(mlvl_scores)
+    results.raw_scores = torch.cat(mlvl_raw_scores) # MODIFY: add raw_scores to results
     results.labels = torch.cat(mlvl_labels)
     if with_score_factors:
         results.score_factors = torch.cat(mlvl_score_factors)
```

You can see the full function [here](https://gist.github.com/Y-T-G/a08c1cc1407699c27ba94bf5c7df9598). I added comments to distinguish what was modified from what wasn't. I am storing the class probabilities in  `results.raw_scores` so that it gets returned with the prediction results. Once you define this new function, you can monkey patch your `model` as follows:

```python
from types import MethodType
model.bbox_head._predict_by_feat_single = MethodType(_predict_by_feat_single, model.bbox_head)
```

The reason we use `MethodType` is because we want the function to have access to the `self` identifier (shout-out to this [SO answer](https://stackoverflow.com/a/64071963/8061030) for this enlightement). If we simply assigned it normally, it will not be able to access `self` and hence throw an error.

## Conclusion

With the new modified function in place, you can now call `inference_detect(model, image)` as usual and it would give you the class probabilities as part of the `pred_instances.raw_scores`. It's that simple.
