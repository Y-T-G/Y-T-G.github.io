---
title: Blur-Anything
seo_title: Blur Anything
summary: Track and blur any object in a video with the click of a button.
description: Blur Anything is an optimized adaptation of Track-Anything (2023), utilizing Meta’s Segment Anything for downstream blurring task in a video.
slug: blur-anything
author: Mohammed Yasin

draft: false
date: 2023-05-04T06:05:00+08:00
lastmod: 
expiryDate: 
publishDate: 

feature_image: 
feature_image_alt: 

project types: 
    - Instance Segmentation

techstack:
    - PyTorch
    - Gradio
    - OpenCV
live_url: https://huggingface.co/spaces/Y-T-G/Blur-Anything
source_url: https://github.com/Y-T-G/Blur-Anything
---

# Blur Anything For Videos

Blur Anything is an adaptation of the excellent [Track Anything](https://github.com/gaomingqi/Track-Anything) project which is in turn based on Meta's Segment Anything and XMem. It allows you to blur anything in a video, including faces, license plates, etc.

<div>
<a src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square" href="https://huggingface.co/spaces/Y-T-G/Blur-Anything">
<img src="https://img.shields.io/badge/%F0%9F%A4%97-Open_in_Spaces-informational.svg?style=flat-square">
</a>
</div>

## Get Started

```shell
# Clone the repository:
git clone https://github.com/Y-T-G/Blur-Anything.git
cd Blur-Anything

# Install dependencies: 
pip install -r requirements.txt

# Run the Blur-Anything gradio demo.
python app.py --device cuda:0
# python app.py --device cuda:0 --sam_model_type vit_b # for lower memory usage
```

## To Do

- [x] Add a gradio demo
- [ ] Add support to use YouTube video URL
- [ ] Add option to completely black out the object

## Acknowledgements

The project is an adaptation of [Track Anything](https://github.com/gaomingqi/Track-Anything) which is based on [Segment Anything](https://github.com/facebookresearch/segment-anything) and [XMem](https://github.com/hkchengrex/XMem).
