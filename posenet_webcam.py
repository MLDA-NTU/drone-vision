import tensorflow as tf
import numpy as np
import cv2

from utils import keypoint_decoder, decode_singlepose

# code from https://medium.com/roonyx/pose-estimation-and-matching-with-tensorflow-lite-posenet-model-ea2e9249abbd

# Pairs represents the lines connected from joints
# e.g. (5,6) is from leftShoulder to rightShoulder
# https://www.tensorflow.org/lite/models/pose_estimation/overview
parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]


def parse_output(scores, offset, threshold):
    '''
    Input:
        heatmap_data - hetmaps for an image. Three dimension array
        offset_data - offset vectors for an image. Three dimension array
        threshold - probability threshold for the keypoints. Scalar value
    Output:
        array with coordinates of the keypoints and flags for those that have
        low probability
    '''

    joint_num = scores.shape[-1]  # number of joints
    pose_kps = np.zeros((joint_num,3), np.int32)   # initialize array

    max_prob = np.max(scores, axis=-1)

    for i in range(joint_num): # loop through the number of joints
        scores_keypoint = scores[:,:,i] # get the probability heatmap for that joint i

        max_prob = np.max(scores_keypoint)
        # this line yields only 1 pose
        # get the max probability position in that layer
        max_idx = np.argmax(scores_keypoint)
        max_pos = np.unravel_index(max_idx, scores_keypoint.shape)
        max_pos = np.array(max_pos)
        # max_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
        
        # scale back the position
        remap_pos = np.array(max_pos/8*257,dtype=np.int32)

        # apply the offset
        pose_kps[i,0] = int(remap_pos[0] + offset[max_pos[0], max_pos[1], i])
        pose_kps[i,1] = int(remap_pos[1] + offset[max_pos[0], max_pos[1], i+joint_num])
        
        max_prob = np.max(scores_keypoint)

        if max_prob > threshold:
            if pose_kps[i,0] < 257 and pose_kps[i,1] < 257:
                pose_kps[i,2] = 1

    return pose_kps

def draw_kps(show_img,kps, ratio=None):
    for i in range(5,kps.shape[0]):
      if kps[i,2]:
        if isinstance(ratio, tuple):
          cv2.circle(show_img,(int(round(kps[i,1]*ratio[1])),int(round(kps[i,0]*ratio[0]))),2,(0,255,255),round(int(1*ratio[1])))
          continue
        cv2.circle(show_img,(kps[i,1],kps[i,0]),2,(0,255,255),-1)
    return show_img

def draw_deviations(img, keypoints, pairs):

  for i, pair in enumerate(pairs):
    cv2.line(img, (keypoints[pair[0]][1], keypoints[pair[0]][0]), (keypoints[pair[1]][1], keypoints[pair[1]][0]), color=(0,255,0), lineType=cv2.LINE_AA, thickness=1)

# SET UP TENSORFLOW LITE
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
cap = cv2.VideoCapture(0)
# Set VideoCaptureProperties 
cap.set(3, 1280)    # width = 1280
cap.set(4, 720)     # height = 720

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
    keypoints = decode_singlepose(heatmaps, offsets, outputStride, 0.5) # list of keypoint, each keypoint is ((y,x), score)

    threshold = 0.3
    for i, keypoint in enumerate(keypoints):
        pos, score = keypoint
        if score < threshold:
            continue    # skip if score is below threshold

        # scale x and y back to original image size
        y = int(round(pos[0] * img.shape[0] / INPUT_HEIGHT))
        x = int(round(pos[1] * img.shape[1] / INPUT_WIDTH))

        cv2.circle(img,(x,y),5,(0,255,0),-1)    # draw keypoint as circle
        cv2.putText(img,keypoint_decoder[i],(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) # put the name of keypoint

    cv2.imshow("pose", img) # show the image with keypoints
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()