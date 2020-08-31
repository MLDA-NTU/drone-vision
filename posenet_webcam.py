import tensorflow as tf
import numpy as np
import cv2

from utils import keypoint_decoder, decode_singlepose, draw_keypoints

# SET UP TENSORFLOW LITE
# ----------------------
model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# shape is (batch_size, height, width, channel)
INPUT_HEIGHT = input_details[0]['shape'][1]
INPUT_WIDTH = input_details[0]['shape'][2]
FLOATING_MODEL = input_details[0]['dtype'] == np.float32

INPUT_INDEX = input_details[0]['index']
HEATMAP_INDEX = output_details[0]['index']
OFFSET_INDEX = output_details[1]['index']


# SET UP WEBCAM
# -------------
cap = cv2.VideoCapture(0)
# Set VideoCaptureProperties 
cap.set(3, 1280)    # width = 1280
cap.set(4, 720)     # height = 720

# MAIN LOOP
# ---------
while True:
    success, img = cap.read()   # read webcam capture
    img_input = cv2.resize(img.copy(), (INPUT_WIDTH, INPUT_HEIGHT)) # resize to fit model's input
    img_input = np.expand_dims(img_input, axis=0)   # add batch dimension

    if FLOATING_MODEL:
        img_input = (np.float32(img_input) - 127.5) / 127.5

    # TensorFlow Lite API
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    interpreter.set_tensor(INPUT_INDEX, img_input)  # load image input to INPUT_INDEX
    interpreter.invoke()    # run the model

    heatmaps = interpreter.get_tensor(HEATMAP_INDEX)    # obtain heatmaps
    offsets = interpreter.get_tensor(OFFSET_INDEX)      # obtain offsets
    heatmaps = np.squeeze(heatmaps) # remove batch dimension
    offsets = np.squeeze(offsets)   # remove batch dimension

    outputStride = 32   # from the model
    keypoints = decode_singlepose(heatmaps, offsets, outputStride)     # list of keypoint, each keypoint is ((y,x), score). see utils.py for implementation

    threshold = 0.2
    scaleX = img.shape[1]/INPUT_WIDTH       # scale back to original image size
    scaleY = img.shape[0]/INPUT_HEIGHT      # scale back to original image size
    draw_keypoints(img, keypoints, threshold=threshold, scaleX=scaleX, scaleY=scaleY)   # see utils.py for implementation

    cv2.imshow("pose", img)     # show the image with keypoints
    if cv2.waitKey(1) & 0xFF == ord('q'):   # terminate window when press q
        break

cv2.destroyAllWindows()