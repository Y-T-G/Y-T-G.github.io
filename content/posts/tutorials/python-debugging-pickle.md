---
title: Debugging Python Functions Using Pickling Trick
seo_title: Debugging Python Functions Using Pickling Trick
summary: A useful trick to debug deeply embedded functions in Python.
description: I show how pickle can be used to easily debug Python functions.
slug: debugging-python-pickling-trick
author: Mohammed Yasin

draft: false
date: 2023-08-12T00:32:00+08:00
lastmod:
expiryDate: 
publishDate: 

feature_image: 
feature_image_alt: 

categories:
  - Tutorials
tags:
  - Python
  - Debugging

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

## Introduction

Being able to debug a program is an essential skill. It is especially important when you are working with a large codebase that has many functions and classes. In such cases, it is not always easy to figure out where the bug is. This is especially true when the bug is in a function that is deeply embedded in the codebase. Rerunning the whole program just to test a specific function that is triggered by a specific action can be tiring and time-consuming.

In this article, I show how the `pickle` module can be used to easily debug such Python functions that are part of a deep call stack of functions.

## How to use `pickle` for debugging?

The `pickle` module allows to dump and load almost any Python object. This is useful when you want to save the state of an object and load it later. Now, how does this help in the case of debugging Python functions?

A function usually takes some inputs to produce some outputs. This means, to be able to call a function, we need to have the required inputs available. When a function is deeply embedded into a codebase, the inputs it receives most likely would have undergone a lot a processing that is difficult replicate by hand. This is where `pickle` comes in. We can use `pickle` to save the processed inputs right before it is passed to the function and then load it in a different session to call the function with the same inputs. This way, we can test the function without having to run the whole program.

Recently, for my [Blur Anything]({{< ref "/projects/blur-anything/index.md" >}}) project, I wanted to integrate MobileSAM with ONNX inference into the existing codebase which was using the vanilla PyTorch inference frontend. The original inference was handled by the `predict()` function responsible for returning the outputs from the SAM model as part of the `BaseSegmenter` class. The `predict()` function looked like this:

```python
 def predict(self, prompts, mode, multimask=True):
    ...
    ...
    ...
    return masks, scores, logits
```

To implement the ONNX runtime inference, I would have to modify this function such that it takes the same inputs and produces the same outputs, so that the rest of the program which utilizes these outputs can continue to function the same, as if nothing had changed. But testing this function by running the whole program isn't easy.

To test the function, I first needed to run the `app.py` script, which first loads the models and initializes all the necessary variables, and then starts the Gradio server. In the Gradio interface, I have to select/upload a video and then click on the `Get video info` button for the program to process the video to obtain the metadata. Only then would the interface allow me to select a point on the video frame to generate the mask, at which point the `predict()` function is called. This whole process can take over 30 seconds each time. So if I wanted to test the function again with a few changes, I would have to repeat the whole process.

Instead of going through all that trouble, I could just import the `pickle` module into the `base_segmenter.py` file where the `predict()` function is located and then use `pickle.dump()` to dump the `prompts` variable by adding the `pickle.dump()` statement as the first line in the function. The `mode` variable needn't be dumped as it is just a `string` that I can set easily:

```python
def predict(self, prompts, mode, multimask=True):
    pickle.dump(prompts, open("prompts.pkl", "wb"))
    ...
    ...
    ...
    return masks, scores, logits
```

With the `prompts` variable dumped, I can now simply start a Jupyter Notebook and load the `prompts` variable from the dumped pickle file as if it had been processed by all the previous parts of the program. Then I can initialize the `BaseSegmenter` class by importing it from the `base_segmenter.py` file where it is located. Now, all I need to do is simply call the `BaseSegmenter.predict()` function with the loaded `prompts` variable and the `mode` variable set to an appropriate value to get outputs that are the same as the ones I would get had I ran the whole program.

By knowing the outputs from the original unmodified function, I can start creating a new function called `predict_onnx()` that uses the ONNX runtime to perform similar inference on the same inputs to produce the same outputs. I don't have to worry about running the whole program to test any of the changes, as I can simply call the function and check the outputs to ensure parity.

Moreover, Python makes it easy to add a new function to an existing instance of a class. I can simply assign the `predict_onnx()` function to the `BaseSegmenter` instance and the function will have access to the `self` property of the instance:

```python
from base_segmenter import BaseSegmenter

segmenter = BaseSegmenter()

def predict_onnx(self, prompts, mode, multimask=True):
    #Add code to perform inference using ONNX runtime
    ...
    ...
    ...
    return masks, scores, logits

segmenter.predict_onnx = predict_onnx()

#prompts that was loaded from the pickle file
segmenter.predict_onnx(prompts, mode='mask')
```

Now all I need to do is make sure the outputs produced by the `predict_onnx()` is correct and equivalent to the one produced by the original `predict()` function. Once I do that, I can add my new function to the `BaseSegmenter` class and rename the old `predict()` function to `predict_pt()`. I will then define `self.predict` to point to either `predict_pt` or `predict_onnx` based on whether I am using the PyTorch or ONNX runtime. This way, I can easily switch between the two runtimes without having to change any other part of the codebase.

And that's it. We have added a function to handle ONNX inference without needing to the run the whole program everytime we wanted to test it. This is just one example of how `pickle` can be used to debug Python functions. I am sure there are many other ways to use it. If you know of any other interesting use cases, please let me know in the comments below.