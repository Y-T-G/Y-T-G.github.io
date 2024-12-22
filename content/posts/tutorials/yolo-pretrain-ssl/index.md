---
title: Pretrain YOLO Backbone Using Self-Supervised Learning With Lightly
seo_title: Pretrain YOLO Backbone Using Self-Supervised Learning With Lightly
summary: Use Lightly to pretrain a YOLO backbone through self-supervised learning and then fine-tune it in Ultralytics.
description: Use Lightly to pretrain a YOLO backbone through self-supervised learning and then fine-tune it in Ultralytics.
slug: yolo-pretrain-ssl
author: Mohammed Yasin

draft: false
date: 2024-12-22T17:07:56+08:00
lastmod: 
expiryDate: 
publishDate: 

feature_image: yolo-pretrain-ssl.png
feature_image_alt: Pretraining YOLO with SSL

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

[LightlySSL](https://github.com/lightly-ai/lightly) is an elegant and easy-to-use framework for self-supervised learning. It allows you effortlessly pretrain a backbone of your choice through several popular self-supervised learning techniques. In this guide, we will look at how to use Lightly to pretrain a YOLO backbone using DINO and also how to load it back into Ultralytics for fine-tuning. The Colab notebook is [here](http://colab.research.google.com/drive/1D1he_wR8AZt3wn-t8aSnR6s4SMV43jhv).

## Implementation

We will modify [this example notebook](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/dino.ipynb) demonstrating how to use DINO through Lightly. There are only a few modifications we need to make. The first and the most obvious one is to change the backbone to the one used by YOLO, which we do as follows:

```python
yolo = YOLO("yolo11n.pt")

class PoolHead(nn.Module):
  """ Apply average pooling to the output."""
  def __init__(self, f, i):
    super().__init__()
    self.f = f  # receive the outputs from these layers
    self.i = i  # layer number
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

  def forward(self, x):
    return self.avgpool(x)

# Only backbone
yolo.model.model = yolo.model.model[:12]  # Keep first 12 layers
yolo.model.model[-1] = PoolHead(yolo.model.model[-1].f, yolo.model.model[-1].i)  # Replace 12th layer with PoolHead
```

In this snippet, we are first loading the YOLO model and then stripping away the head. Then we attach a `PoolHead` to the backbone. This `PoolHead` would take the output of the previous layer and apply adaptive average pooling to reduce the spatial dimensions of the feature map to a fixed and consistent size (`1x1`). This is required because the spatial dimensions would otherwise vary depending on the size of the input which would make it difficult to attach the DINO head to the backbone since it requires a fixed input size.

After that, we perform a dummy forward pass to get the output channel size of the backbone:

```python
CROP_SIZE = 224
dummy = torch.rand(2, 3, CROP_SIZE, CROP_SIZE)
out = yolo.model(dummy)
input_dim = out.flatten(start_dim=1).shape[1]
```

The `input_dim` in this case is `256` and we use that along with the YOLO backbone to initialize the `DINO` model:

```python
backbone = yolo.model.requires_grad_()
backbone.train()
model = DINO(backbone, input_dim)
```

Here, we also do two other things prior to creating the model. We enable gradient calculation for the backbone which is disabled by default in Ultralytics. And we also put the backbone in training mode so that BatchNorm statistics get updated during training.

And lastly, we create a `transform` with the default `mean` and `std` used by YOLO to keep it consistent with what's used by YOLO:

```python
normalize = dict(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))  # YOLO uses these values
transform = DINOTransform(global_crop_size=CROP_SIZE, normalize=normalize)
```

The `CROP_SIZE` is `224` by default. You could use a different size such as `640` which is more consistent with the default image size in YOLO, but it would also consume more VRAM during training. There's also `local_crop_size` that you can control, which is by default set to `96`. These are all DINO related parameters and you can read about them in the [Lightly Docs](https://docs.lightly.ai/self-supervised-learning/lightly.transforms.html#lightly.transforms.dino_transform.DINOTransform).

And that's pretty much all the modifications you need to make. You then simply load your dataset, create your dataloader, define the loss function and optimizer, and start training. I am just using the defaults in the DINO notebook.

## Loading The Pretrained Backbone in Ultralytics

To load the pretrained backbone back into Ultralytics after pretraining, you just need these few lines:

```python
from ultralytics import YOLO

# Load the same model that was used for pretraining
yolo = YOLO("yolo11n.pt")

# Transfer weights from pretrained model
yolo.model.load(model.student_backbone)

# Save the model for later use
yolo.save("pretrained.pt")
```

This snippet transfers the weights from the matching layers in the pretrained backbone back to the loaded YOLO model. And then you can just save it as a typical Ultralytics model and load it normally for fine-tuning:

```python
from ultralytics import YOLO


yolo = YOLO("pretrained.pt")

# Nothing different from your usual process
results = yolo.train(data="VOC.yaml", epochs=1, freeze=0, warmup_epochs=0, imgsz=640, val=False)
```

## Results From Fine-Tuning

Pretraining is usually performed on a very large dataset and for several epochs. Nevertheless, I tried performing a sanity check by fine-tuning the pretrained backbone on the same PASCAL VOC dataset in Ultralytics that was also used for pretraining. The performance was not better than starting from the COCO pretrained model in this case, but then again, like I said, this was just a sanity check and not actual pretraining which takes much longer.

One epoch of fine-tuning with SSL pretrained backbone:
```python
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/1      3.25G      1.562      3.229      1.729         54        640: 100%|██████████| 1035/1035 [06:15<00:00,  2.76it/s]
            Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 155/155 [00:52<00:00,  2.97it/s]
            all       4952      12032      0.265      0.234      0.162     0.0887
```

One epoch of fine-tuning with COCO pretrained backbone:
```python
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
1/1      2.94G      1.169      2.335      1.419         54        640: 100%|██████████| 1035/1035 [06:57<00:00,  2.48it/s]
            Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 155/155 [00:51<00:00,  3.03it/s]
            all       4952      12032      0.585      0.537      0.556      0.356
```

## Conclusion

This was a short guide on how to use Lightly to pretrain a YOLO backbone and then load it back into Ultralytics for fine-tuning. Unfortunately, I didn't have the resources to run longer and more thorough experiments to check the difference it makes. You can also check out [this thread](https://discord.com/channels/752876370337726585/752877278152622100/1306975369114550273) in Lightly Discord that discusses pretraining a YOLO backbone and the caveats.

If you do get better results with pretraining as opposed to starting from COCO pretrained models, you can share the results in the comments. Thanks for reading.