# drone-vision

[pose_matching_with_PoseNet.ipynb](pose_matching_with_PoseNet.ipynb) contains original code from from https://medium.com/roonyx/pose-estimation-and-matching-with-tensorflow-lite-posenet-model-ea2e9249abbd

Run `posenet_webcam.py` to see posenet output with webcam. `posenet.py` provides a wrapper around TensorFlow Lite to use PoseNet easier

## Usage

```python
from posenet import PoseNet

posenet = PoseNet("posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
img_input = posenet.prepare_input(img)  # img is an image (height, width, 3)

# get keypoints for single pose estimation. it is a list of 17 keypoints
keypoints = posenet.predict_singlepose(img_input)

# draw keypoints to the original image
posenet.draw_keypoints_to_image(img, keypoint)
```

Check `posenet_webcam.py` for a sample code using PoseNet wrapper with webcam input.

Currently only single-pose estimation is supported.