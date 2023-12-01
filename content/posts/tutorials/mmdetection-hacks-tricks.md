---
title: MMDetection 3.x Hacks and Tricks
seo_title: MMDetection 3.x Hacks and Tricks
summary: Some useful tricks and hacks I learnt while using MMDetection for my research work.
description: A guide showing how to reinitalize the model, filter the dataset and more in MMDetection.
slug: mmdetection-hacks-tricks
author: Mohammed Yasin

draft: false
date: 2023-12-01T19:18:02+08:00
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
series:
  - MMDetection

toc: true
related: true
social_share: false
newsletter: true
disable_comments: false
---

## Introduction

MMDetection, and in fact, all of the tools provided by OpenMMLab are all  useful for both research-related work and production. I have shown how to get prediction probabilities in MMDetection in one of my past tutorials by monkey patching some functions, and in this guide, I will be covering some more small but useful tricks I learnt while meddling with it for my research work.

## Runner - the place where everything comes together

If you go through the `train.py` file in MMDetection, you'll notice that the code ultimately culminates in the creation of a `Runner` object. The `Runner` object is where MMDetection, technically MMEngine, sets everything up, usually by parsing the config file that is passed to the script. It is also the object that stores the initialized model and exposes the method to initiate the training. It is often useful to initialize the `Runner` object once because every time it is created, it has to go through the process of setting everything up once more, which can slow down your workflow significantly. The best way is probably to start up a Jupyter Notebook and create the runner in it. Then you can reuse the runner without having to reinitialize it every time.

## Reinitializing the model

Let's say you created a `Runner` object and then called the `train()` method to train it. After the training finishes, the trained model would be stored as a property of the runner object `runner.model`. You can use this model in the same way you can use a loaded pre-trained model in MMDetection. But what if you want to reinitialize the model? How do you do it without reinitializing the runner?

I wrote a small function just to do that:

```python
def init_model(runner):
    """
    Initializes a new model for the next training cycle.
    Args:
        runner (Runner): Runner object.
    Returns:
        runner (Runner): Runner object with initialized model.
    """
    
    # Reset the optimizer and scheduler
    runner.optim_wrapper = runner.cfg.optim_wrapper
    runner.param_schedulers = runner.cfg.param_scheduler

    # Build a new model
    runner.model = runner.build_model(runner.cfg.model)
    runner.model = runner.wrap_model(
                runner.cfg.get('model_wrapper_cfg'), runner.model)

    return runner
```

You can pass your `runner` to the above function and it would reinitialize the model and return the updated `runner` with the reinitialized model. In the function, we also reset the optimizer and scheduler, as otherwise the training would throw an error.

## Filtering the dataset

The `Runner` object also gives you access to loaded dataset. And this allows us to do another nifty hack to filter the dataset. This was useful for me when I wanted to implement active learning to create datasets for each cycle.

The dataset object in `runner.train_dataloader.dataset` has a property called `_indices`.  This property is used to filter the dataset during initialization of the dataset. When the `Runner` object is created, it doesn't initialize the dataset right away. MMEngine implements lazy-loading that delays the initialization of the dataset until the dataset is used, which is when the training begins. So you could use the `_indices` property to pass the indices of the data you want to keep in the final dataset that would be used during training. However, there's one issue. The implementation is currently broken. So we have to again resort to monkey-patching. Specifically, we patch the `full_init()` method using the following redefinition:

```python
def full_init(self) -> None:
    """Load annotation file and set ``BaseDataset._fully_initialized`` to
    True.

    If ``lazy_init=False``, ``full_init`` will be called during the
    instantiation and ``self._fully_initialized`` will be set to True. If
    ``obj._fully_initialized=False``, the class method decorated by
    ``force_full_init`` will call ``full_init`` automatically.

    Several steps to initialize annotation:

        - load_data_list: Load annotations from annotation file.
        - load_proposals: Load proposals from proposal file, if
            `self.proposal_file` is not None.
        - filter data information: Filter annotations according to
            filter_cfg.
        - slice_data: Slice dataset according to ``self._indices``
        - serialize_data: Serialize ``self.data_list`` if
        ``self.serialize_data`` is True.
    """
    if self._fully_initialized:
        return
    # load data information
    self.data_list = self.load_data_list()
    # get proposals from file
    if self.proposal_file is not None:
        self.load_proposals()

    # MODIFIED: Filter indices before filtering data to maintain indice sanity

    # Get subset data according to indices.
    if self._indices is not None:
        self.data_list = self._get_unserialized_subset(self._indices)

    # filter illegal data, such as data that has no annotations.
    self.data_list = self.filter_data()

    # serialize data_list
    if self.serialize_data:
        self.data_bytes, self.data_address = self._serialize_data()

    self._fully_initialized = True
```

In this implementation, we filter the indices before the `filter_data()` method is called which is used to remove invalid data or images with no annotations. If we don't use the indices to filter the data before that, the indices that we passed would be pointing to the wrong data and hence mess up the whole thing. You can monkey-patch the method with the above definition through `MethodType`:

```python
def patch_dataset_init(dataset):
    dataset.full_init = MethodType(full_init, dataset)
    return dataset
```

I can then use the above in the following function to filter the dataset and reload the dataloader:

```python
def filter_dataset(runner, indices):
    """
    Filter dataset with indices.

    Args:
        runner (Runner): Runner object.

    Returns:
        runner (Runner): Runner object with initialized model.
    """

    # Modify the dataset to only include the labeled images
    # `full_init()` filters the dataset based on `_indices`
    dataset = runner.train_dataloader.dataset
    dataset = patch_dataset_init(dataset)
    dataset._indices = indices
    dataset._fully_initialized = False
    dataset.full_init()

    # Build a new dataloader with the modified dataset
    dataloader_cfg = runner.cfg.train_dataloader.copy()
    dataloader_cfg['dataset'] = dataset
    runner._train_loop.dataloader = runner.build_dataloader(dataloader_cfg)

    return runner

```

In the above function, we do a bunch of things. Firstly, we patch the `full_init()` method. Then we mark the dataset as uninitialized by setting the `_fully_initialized` property as `False`. If we don't do that, the `full_init()` will not reinitialize the dataset because it would consider it already initialized and hence skip the filtering process altogether. We also built the dataloader again with the updated dataset. Otherwise, the training will continue to use the old dataset.

## Reinitializing the visualizer

One thing that you might also want to do when you're doing something like active learning is to start a new visualizer session. Visualizer is what MMEngine uses to log the metrics during training. So if you want each cycle of your active learning to use a new and separate visualizer session, especially if you're using something like Weights & Biases for visualization, you can reset them by first closing the previous session and starting a new one:

```python
# Close visualization session
runner.visualizer.close()
runner.visualizer.closed = True
```

We also create a new `closed` property in the visualizer object to store the status of the visualizer. It's useful for checking whether the visualizer has been closed already. I wasn't able to find a native method to check this, so this will have to do. We create a new visualizer session by running the following:

```python
# Create new visualizer session
if hasattr(runner.visualizer, 'closed') and runner.visualizer.closed:
 for vis_backend in runner.visualizer._vis_backends.values():
  vis_backend._init_env()
  vis_backend.add_config(runner.cfg)
  runner.visualizer.closed = False
```

This would create a new session so that your previous training session is separated from the next one.

## Useful properties

Besides the above, there are some other useful properties of `runner` object that you can modify or make use of:

1. **work_dir**: This is the path to the directory where the log folders are created each time the runner is initialized. You can't modify the `runner.work_dir` directly. Instead, you have to modify `runner._work_dir`.
2. **log_dir**: As the name suggests, this is the subfolder where the training session will store the visualization data, the logs and also the checkpoints. Similar to `work_dir`, you have to modify `runner._log_dir` instead of `runner.log_dir`.
3. **cfg**: The `config` is stored in the `runner.cfg` property. We have already used this before in our functions above.
4. **data_list**: In `runner.train_dataloader.dataset.data_list`, you can find the list of all images along with their annotation information as parsed during initialization. This is the list that you would refer to to determine the indices of the data you want to keep during the filtering process.

## Extra: `importlib.reload`

We have seen how you can reinitialize the model through the `init_model()` function above. But what if you made some changes to your model file? Reinitializing the model would not cause those changes to take effect in the Jupyter notebook. To make the changes take effect without restarting the notebook kernel altogether, you can make use of `importlib.reload` to reload the model file. For example:

```python
from importlib import reload
import mmdet.models.detectors.retinanet
reload(mmdet.models.detectors.retinanet)
```

Doing the above would cause the changes in the `mmdet/models/detectors/retinanet.py` file to take effect so that if we reinitialize the model and if the config is using RetinaNet, it would initialize it with the updated model.

## Conclusion

And that's all I have to offer in this guide. I may find some more useful tricks as I dig deeper, but I hope the ones I highlighted here come in handy. Thanks for reading!
