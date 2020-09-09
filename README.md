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

## Download and convert PoseNet ResNet float model

Download code from this repo: https://github.com/atomicbits/posenet-python

```
git clone https://github.com/atomicbits/posenet-python.git
cd posenet-python
```

Installing dependencies of this package will mess up tensorflow-gpu install. Thus setting up a virtual environment is needed. Instruction below is for `conda`

```
conda create -n convert-tfjs python=3.8
conda activate tfjs
conda install pip
pip install -r requirements.txt
```

Run the sample script to trigger the download and model conversion process

```
python image_demo.py --model resnet50 --stride 16 --image_dir ./images --output_dir ./output
```

The converted ResNet50 model is now saved at `posenet-python/_tf_models/posenet/resnet50_float/stride16`. It contains the following

```
saved_model.pb
variables/
```

Copy the content of this folder to another folder for easy access to the converted model.