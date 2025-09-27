---
title: Pruning Ultralytics YOLO Models with NVIDIA Model Optimizer
seo_title: Pruning Ultralytics YOLO Models with NVIDIA Model Optimizer
summary: Prune Ultralytics YOLO models with NVIDIA Model Optimizer and then fine-tune it using Ultralytics for improved accuracy and performance.
description: Prune Ultralytics YOLO models with NVIDIA Model Optimizer and then fine-tune it using Ultralytics for improved accuracy and performance.
slug: yolo-prune
author: Mohammed Yasin

draft: false
date: 2025-09-27T17:40:47+08:00
lastmod: 
expiryDate: 
publishDate: 

feature_image: yolo-prune.png
feature_image_alt: pruning yolo models

categories:
  - Tutorials
tags:
  - YOLO
  - Object Detection
series:
  - NVIDIA Model Optimizer

toc: true
related: true
social_share: true
newsletter: false
disable_comments: false
---

## Introduction

<a href="https://colab.research.google.com/drive/1BX6zAm1Q_VNR8iXfaq0Aoctpw2NaTZAu" target="_blank" style="text-decoration:none;">
  <div style="display:inline-flex;align-items:center;padding:6px 12px;background-color:#F9AB00;border-radius:4px;color:#000;font-family:'Roboto','Helvetica','Arial',sans-serif;font-size:14px;font-weight:500;">
    <svg viewBox="0 0 24 24" width="20" height="20" style="margin-right:8px;">
      <g id="colab-logo">
        <path d="M4.54,9.46,2.19,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36A3.59,3.59,0,0,1,4.54,9.46Z" style="fill:var(--colab-logo-dark)"></path>
        <path d="M2.19,7.1,4.54,9.46a3.59,3.59,0,0,1,5.08,0l1.71-2.93h0l-.1-.08h0A6.93,6.93,0,0,0,2.19,7.1Z" style="fill:var(--colab-logo-light)"></path>
        <path d="M11.34,17.46h0L9.62,14.54a3.59,3.59,0,0,1-5.08,0L2.19,16.9a6.93,6.93,0,0,0,9,.65l.11-.09" style="fill:var(--colab-logo-light)"></path>
        <path d="M12,7.1a6.93,6.93,0,0,0,0,9.79l2.36-2.36a3.59,3.59,0,1,1,5.08-5.08L21.81,7.1A6.93,6.93,0,0,0,12,7.1Z" style="fill:var(--colab-logo-light)"></path>
        <path d="M21.81,7.1,19.46,9.46a3.59,3.59,0,0,1-5.08,5.08L12,16.9A6.93,6.93,0,0,0,21.81,7.1Z" style="fill:var(--colab-logo-dark)"></path>
      </g>
    </svg>
    Open in Colab
  </div>
</a>

[NVIDIA Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (ModelOpt) is a toolkit by NVIDIA that provides model optimization strategies through an easy-to-use API. It minimizes the code changes needed to apply state-of-the-art optimizations by using a wrapper-based approach that preserves the model's attributes while performing optimization behind the scenes. In this guide, we will prune and fine-tune the Ultralytics YOLOv8m model using NVIDIA's Model Optimizer and show how simple the process is. The Colab notebook with the full code can be accessed by clicking the "Open in Colab" badge above.

## Modifying Model Saving and Loading Mechanism

One hurdle with using ModelOpt is that the wrapped model can't be pickled. This is problematic because Ultralytics relies on pickling the entire model object when saving. To resolve this, I created a custom branch that simplifies saving and loading models optimized with ModelOpt, which you can install as follows:

```bash
pip install nvidia-modelopt git+https://github.com/ultralytics/ultralytics@qat-nvidia
```

In this branch, I have modified the `save_model()` method of `BaseTrainer` to save the model's `state_dict`, YAML config and the ModelOpt config instead of pickling the whole model:

```diff
@@ -572,12 +581,26 @@ class BaseTrainer:

         # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
         buffer = io.BytesIO()
+        model = deepcopy(unwrap_model(self.ema.ema))
+        extras = {}
+        if hasattr(model, "_modelopt_state"):
+            import modelopt.torch.opt as mto
+
+            extras = {  # model can't be pickled; saving state_dict
+                "modelopt_state": mto.modelopt_state(model),
+                "state_dict": model.state_dict(),
+                "model_class": model.__class__,
+                "yaml": model.yaml,
+                "names": model.names,
+                "nc": model.nc,
+            }
+
         torch.save(
             {
                 "epoch": self.epoch,
                 "best_fitness": self.best_fitness,
                 "model": None,  # resume and final checkpoints derive from EMA
-                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
+                "ema": None if self.args.int8 else model.half(),
                 "updates": self.ema.updates,
                 "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                 "scaler": self.scaler.state_dict(),
@@ -594,6 +617,7 @@ class BaseTrainer:
                 },
                 "license": "AGPL-3.0 (https://ultralytics.com/license)",
                 "docs": "https://docs.ultralytics.com",
+                **extras,
             },
             buffer,
         )
```

This allows us to then rebuild the model using the YAML and restore the state dict and ModelOpt config in the `load_checkpoint()` function:

```diff
@@ -1500,7 +1501,22 @@ def load_checkpoint(weight, device=None, inplace=True, fuse=False):
     """
     ckpt, weight = torch_safe_load(weight)  # load ckpt
     args = {**DEFAULT_CFG_DICT, **(ckpt.get("train_args", {}))}  # combine model and default args, preferring model args
-    model = (ckpt.get("ema") or ckpt["model"]).float()  # FP32 model
+    model = ckpt.get("ema") or ckpt["model"]
+
+    if "modelopt_state" in ckpt:  # ModelOpt model
+        import modelopt.torch.opt as mto
+
+        # rebuild from YAML
+        model = ckpt["model_class"](ckpt["yaml"], verbose=False)
+        model.names = ckpt["names"]
+        model.nc = ckpt["nc"]
+        model.yaml = ckpt["yaml"]
+        # restore model and ModelOpt config
+        with torch.no_grad():
+            mto.restore_from_modelopt_state(model, ckpt["modelopt_state"])
+            model.load_state_dict(ckpt["state_dict"])
+
+    model = model.float()

     # Model compatibility updates
     model.args = args  # attach args to model
```

## Creating the Trainer

To apply pruning using ModelOpt, we create a custom trainer class that inherits from the original trainer and overrides `_setup_train()` to prune the model before training begins. This way, training fine-tunes the already pruned model.

```python
class PrunedTrainer(model.task_map[model.task]["trainer"]):
  def _setup_train(self):
    """Modified setup model that adds pruning."""
    from ultralytics.utils import LOGGER
    from ultralytics.utils.torch_utils import ModelEMA
    import torch, math

    super()._setup_train()

    def collect_func(batch):
        return self.preprocess_batch(batch)["img"]

    def score_func(model):
        model.eval()
        self.validator.args.save = False
        self.validator.args.plots = False
        self.validator.args.verbose = False
        self.validator.is_coco = False
        metrics = self.validator(model=model)
        self.validator.args.save = self.args.save
        self.validator.args.plots = self.args.plots
        self.validator.args.verbose = self.args.verbose
        return metrics["fitness"]

    prune_constraints = {"flops": "66%"}  # prune to 66% of original FLOPs

    self.model.is_fused = lambda: True  # disable fusing

    self.model, prune_res = mtp.prune(
        model=self.model,
        mode="fastnas",
        constraints=prune_constraints,
        dummy_input=torch.randn(1, 3, self.args.imgsz, self.args.imgsz).to(self.device),
        config={
            "score_func": score_func,  # scoring function
            "checkpoint": "modelopt_fastnas_search_checkpoint.pth",  # saves checkpoint during subnet search
            "data_loader": self.train_loader,  # training dataloader
            "collect_func": collect_func,  # preprocessing function
            "max_iter_data_loader": 20,  # 50 is recommended, but requires more RAM
        },
    )

    self.model.to(self.device)
    self.ema = ModelEMA(self.model)  # wrap EMA
    
    # recreate optimizer and scheduler
    weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
    iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
    self.optimizer = self.build_optimizer(
        model=self.model,
        name=self.args.optimizer,
        lr=self.args.lr0,
        momentum=self.args.momentum,
        decay=weight_decay,
        iterations=iterations,
    )
    self._setup_scheduler()
    LOGGER.info("Applied pruning")
```

Here, `score_func` evaluates model fitness during pruning, and `collect_func` preprocesses each batch. Fusing is disabled since it interferes with subnet search. The key line is `prune_constraints = {"flops": "66%"}`, which defines the pruning target relative to original FLOPs. After pruning, the trainer swaps in the pruned model and rebuilds the optimizer and scheduler.

Training can now be launched with the `PrunedTrainer`:

```python
results = model.train(data="coco128.yaml", trainer=PrunedTrainer, epochs=50)
```

This example fine-tunes the pruned model for 50 epochs on the smaller COCO128 dataset as demonstration. In practice, you should use your actual dataset and a higher epoch count, closer to the original training, for better results. During training, the output logs will show if your constraints can be met:

```
                              Profiling Results                              
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Constraint   ┃ min          ┃ centroid     ┃ max          ┃ max/min ratio ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ flops        │ 17.97G       │ 25.99G       │ 39.47G       │ 2.20          │
│ params       │ 11.61M       │ 17.63M       │ 25.87M       │ 2.23          │
└──────────────┴──────────────┴──────────────┴──────────────┴───────────────┘
                                              
            Constraints Evaluation            
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃              ┃              ┃ Satisfiable  ┃
┃ Constraint   ┃ Upper Bound  ┃ Upper Bound  ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ flops        │ 26.05G       │ True         │
└──────────────┴──────────────┴──────────────┘
```

If your target is below the `min` shown above, pruning will fail. In this case, the minimum achievable FLOPs is 45.5% of the maximum. NVIDIA recommends choosing a constraint near the `centroid`, here about 66%.

## Pruning Results

After training, loading the pruned model and running `model.info()` confirms the FLOPs are reduced to 66%:

```python
>>> pruned_model = YOLO("runs/detect/train/weights/best.pt")
>>> pruned_model.info()
Model summary: 169 layers, 16,369,472 parameters, 16,369,472 gradients, 52.3 GFLOPs
```

The original YOLOv8m FLOPs were:

```python
YOLOv8m summary: 169 layers, 25,902,640 parameters, 0 gradients, 79.3 GFLOPs
```

The pruned model is also smaller: 62.7MB vs. 99MB (37% smaller). To test the speed, I converted the models to TensorRT FP16 engines and ran validation on COCO128. The original model had 6.4ms inference time vs. 5.4ms for the pruned model on an NVIDIA T4. It is not a significant reduction, but still noticeable because in terms of FPS, inference with pruned model is almost higher by 30FPS.

## Conclusion

In this guide, we used NVIDIA Model Optimizer to prune the YOLOv8m model to 66% of its original FLOPs and reduced model size by 37%, with a slight boost in inference speed. Note that to load the `.pt` files from this training, you need to either use the [custom branch described in the beginning](#modifying-model-saving-and-loading-mechanism), or export it to a different format. To learn more about ModelOpt's prune function and the available configurations, refer to [ModelOpt Pruning Docs](https://nvidia.github.io/TensorRT-Model-Optimizer/guides/3_pruning.html) and [GitHub examples](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/pruning).

Thanks for reading.
