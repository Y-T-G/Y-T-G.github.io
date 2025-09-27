---
title: Boosting Inference FPS With Tracker Interpolated Detections
seo_title: Boosting Inference FPS With Tracker Interpolated Detections
summary: Boost your inference performance by exploiting Kalman filter predictions to interpolate detections for skipped frames.
description: Boost your inference performance by skipping inference and exploiting Kalman filter predictions to interpolate detections on intermediate frames.
slug: yolo-tracker-interpolate
author: Mohammed Yasin

draft: false
date: 2024-11-22T00:49:02+08:00
lastmod: 
expiryDate: 
publishDate: 

feature_image: tracker-interpolation.png
feature_image_alt: interpolating detections with tracker

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

<a href="https://colab.research.google.com/drive/1J4fo7Xs2HX4uzaI8C6QBuYorKYeRNqyG?usp=sharing" target="_blank" style="text-decoration:none;">
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

DeepStream has this cool feature where it lets you run inference on every *n*th frame while compensating the rest with predictions from the tracker. It's a pretty simple concept. In this guide, we'll replicate the same feature in `ultralytics`. The Colab notebook contains the full code.

## Retrieving The Predictions From The Tracker

Most trackers integrate Kalman filter to make the object tracking smoother. Kalman filter essentially tries to model the movement of the tracked objects so that it can predict the next possible coordinates of the object. This prediction is typically combined with the coordinates obtained from the detector to get an "average" estimate of the object's location. This is done to make the tracking smoother and more stable. However, we can also use these estimates to interpolate detections for the next frame without having to run inference on that frame. This lets us save compute as we would no longer need to run inference on every frame.

To do this in `ultrlaytics`, we use this snippet of code:

```python
def interpolate(model, frame, path):
    tracker = model.predictor.trackers[0]
    tracks = [t for t in tracker.tracked_stracks if t.is_activated]
    # Apply Kalman filter to get predicted locations
    tracker.multi_predict(tracks)
    tracker.frame_id += 1
    boxes = [np.hstack([t.xyxy, t.track_id, t.score, t.cls]) for t in tracks]
    # Update frame_id in tracks
    for t in tracks:
        t.frame_id = tracker.frame_id
    return Results(frame, path, model.names, np.array(boxes))
```

It's pretty simple. It retrieves the tracked objects that are stored in the tracker, and then advances them to the next state through the Kalman filter estimates. This gives us the next predicted location of the objects, which we then turn into a `Results` object so that it behaves like the output you get from the detector in `ultralytics`

We can integrate this into our pipeline like so:

```python
def infer_on_video(model, filename, output, stride=3, start_frame=5):
  cap = cv2.VideoCapture(filename)

  frame_id = 1
  path = None

  start = time.time()
  while True:
      ret, frame = cap.read()
      if not ret:
          break

      # Interpolate if we reach start_frame and the current frame is not divisible by stride
      if frame_id % stride != 0 and frame_id >= start_frame:
          result = interpolate(model, frame, path)
      else:
          result = model.track(frame, persist=True, verbose=False, classes=[0], iou=0.9)[0]
          # We need this to create Results object manually
          if path is None:
              path = result.path

      frame_id += 1

  cap.release()
```

Here, `stride=3` means that the detector would only be run on every 3rd frame. The other two frames would be interpolated using the Kalman filter predictions.

And this is the result:

<p align="center">
  <img src="https://github.com/Y-T-G/Yasins-Keep/releases/download/v0.0.1/tracker-interpolate-comparison.webp"
  alt="comparison of FPS with and without interpolaton"/>
</p>

The top video shows the predictions without interpolation, while the one below is with interpolation using `stride=3`. The latter has higher FPS as we only infer on every third frame.

## Conclusion

In this short guide, we looked at how we can interpolate detections using the tracker. Using this, we can only infer on every *n*th frame, saving us some compute while increasing the FPS. The caveat is that a smaller stride would give you more precise detections but slower inference performance, while a larger stride would save you more compute but the detections would become more imprecise as it would be relying on estimates more heavily.

Thanks for reading.
