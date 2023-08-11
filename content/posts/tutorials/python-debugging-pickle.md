---
title: Debugging Python Functions Using Pickling Trick
seo_title: Debugging Python Functions Using Pickling Trick
summary: A useful trick to debug deeply embedded functions and methods in Python.
description: I show how pickle can be used to easily debug Python functions and methods.
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

Being able to debug a program is an essential skill. It is especially important when you are working with a large codebase that has many functions, methods and classes. In such cases, it is not always easy to figure out where the bug is. This is especially true when the bug is in a function or a method that is deeply embedded in the codebase. Rerunning the whole program just to test a specific function that is triggered by a specific action can be tiring and time-consuming.

In this post, I show how the `pickle` module can be used to easily debug such Python functions that are part of a deep call stack of functions.

## How to use `pickle` for debugging?

The `pickle` module allows to dump and load almost any Python object. This is useful when you want to save the state of an object and load it later. Now, how does this help in the case of debugging Python functions or methods?

A function or a method usually takes some inputs to produce some outputs. This means, to be able to call a function, we need to have the required inputs available. When a function is deeply embedded into a codebase, the inputs it receives most likely would have undergone a lot of processing that is difficult to replicate by hand. This is where `pickle` comes in. We can use `pickle` to save the processed inputs right before it is passed to the function and then load them in a different session. We can then call the function with the same inputs. This way, we can test the function without having to run the whole program.

## An Example

Recently, for my [Blur Anything]({{< ref "/projects/blur-anything/index.md" >}}) project, I wanted to integrate MobileSAM with ONNX inference into the existing codebase which was using the vanilla PyTorch inference frontend. The original inference was handled by the `predict()` method of the `BaseSegmenter` class and was responsible for returning the outputs from the SAM model. The `predict()` method looked like this:

```python
 def predict(self, prompts, mode, multimask=True):
    ...
    ...
    ...
    return masks, scores, logits
```

To implement the ONNX runtime inference, I would have to modify this method such that it takes the same inputs and produces the same outputs so that the rest of the program which utilizes these outputs can continue to function the same as if nothing had changed. But testing this method by running the whole program isn't easy.

To test the method, I first needed to run the `app.py` script, which first loads the models and initializes all the necessary variables, and then starts the Gradio server. In the Gradio interface, I have to select/upload a video and then click on the `Get video info` button for the program to process the video to obtain the metadata. Only then would the interface allow me to select a point on the video frame to generate the mask, at which point the `predict()` method is called. This whole process can take over 30 seconds each time. So if I wanted to test the method again with a few changes, I would have to repeat the whole process.

Instead of going through all that trouble, I could just import the `pickle` module into the `base_segmenter.py` file where the `predict()` method is located and then use `pickle.dump()` to dump the `prompts` variable by adding the `pickle.dump()` statement as the first line in the method. The `mode` variable needn't be dumped as it is just a `string` that I can set easily:

```python
def predict(self, prompts, mode, multimask=True):
    pickle.dump(prompts, open("prompts.pkl", "wb"))
    ...
    ...
    ...
    return masks, scores, logits
```

With the `prompts` variable dumped, I can now simply start a Jupyter Notebook and load the `prompts` variable from the dumped pickle file as if it had been processed by all the previous parts of the program. Then I can initialize the `BaseSegmenter` class by importing it from the `base_segmenter.py` file where it is located. Now, all I need to do is simply call the `BaseSegmenter.predict()` method with the loaded `prompts` variable and the `mode` variable set to an appropriate value to get outputs that are the same as the ones I would get had I ran the whole program.

By knowing the outputs from the original unmodified method, I can start creating a new function called `predict_onnx()` that uses the ONNX runtime to perform similar inference on the same inputs to produce the same outputs. I don't have to worry about running the whole program to test any of the changes, as I can simply call the function and check the outputs to ensure parity.

Moreover, Python makes it easy to add a new function as a method to an existing object of a class. I can simply assign the `predict_onnx` function to the `BaseSegmenter` instance and the method will have access to the `self` property of the instance:

```python
from base_segmenter import BaseSegmenter

segmenter = BaseSegmenter()

def predict_onnx(self, prompts, mode, multimask=True):
    #Add code to perform inference using ONNX runtime
    ...
    ...
    ...
    return masks, scores, logits

segmenter.predict_onnx = predict_onnx

#prompts that was loaded from the pickle file
segmenter.predict_onnx(prompts, mode='mask')
```

Now all I need to do is make sure the outputs produced by the `predict_onnx()` are correct and equivalent to the one produced by the original `predict()` method. Once I do that, I can add my new method to the `BaseSegmenter` class and rename the old `predict()` method to `predict_pt()`. I will then define `self.predict` to point to either `predict_pt` or `predict_onnx` based on whether I am using the PyTorch or ONNX runtime. This way, I can easily switch between the two runtimes without having to change any other part of the codebase.

## Conclusion

And that's it. We have added a method to handle ONNX inference without needing to run the whole program every time we wanted to test it. This is just one example of how `pickle` can be used to debug Python functions or methods. I am sure there are many other ways to use it. If you know of any other interesting use cases, please let me know in the comments below.
