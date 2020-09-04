import heapq
import tensorflow as tf
import numpy as np
import cv2

keypoint_decoder = [
    "nose",             # 0
    "leftEye",          # 1
    "rightEye",         # 2
    "leftEar",          # 3
    "rightEar",         # 4
    "leftShoulder",     # 5
    "rightShoulder",    # 6
    "leftElbow",        # 7
    "rightElbow",       # 8
    "leftWrist",        # 9
    "rightWrist",       # 10
    "leftHip",          # 11
    "rightHip",         # 12
    "leftKnee",         # 13
    "rightKnee",        # 14
    "leftAnkle",        # 15
    "rightAnkle",       # 16
]

keypoint_encoder = {x: i for i,x in enumerate(keypoint_decoder)}

# Pairs represents the lines connected from joints
# e.g. (5,6) is from leftShoulder to rightShoulder
# https://www.tensorflow.org/lite/models/pose_estimation/overview
parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

# define the skeleton. code from Google's tfjs-models
# each tuple is (parent, child)
poseChain = [
  ('nose',          'leftEye'), 
  ('leftEye',       'leftEar'), 
  ('nose',          'rightEye'),
  ('rightEye',      'rightEar'), 
  ('nose',          'leftShoulder'),
  ('leftShoulder',  'leftElbow'), 
  ('leftElbow',     'leftWrist'),
  ('leftShoulder',  'leftHip'), 
  ('leftHip',       'leftKnee'),
  ('leftKnee',      'leftAnkle'), 
  ('nose',          'rightShoulder'),
  ('rightShoulder', 'rightElbow'), 
  ('rightElbow',    'rightWrist'),
  ('rightShoulder', 'rightHip'), 
  ('rightHip',      'rightKnee'),
  ('rightKnee',     'rightAnkle')
]
parentChildrenTuples = [(keypoint_encoder[parent], keypoint_encoder[child]) for (parent, child) in poseChain]
parentToChildEdges = [childId for (_, childId) in parentChildrenTuples]
childToParentEdges = [parentId for (parentId, _) in parentChildrenTuples]

class PoseNet():
    def __init__(self, model_path):
        ''' initialize TensorFlow Lite with the given model_path '''
        # set up tensorflow lite
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # shape is (batch_size, height, width, channel)
        self.INPUT_HEIGHT = self.input_details[0]['shape'][1]
        self.INPUT_WIDTH = self.input_details[0]['shape'][2]
        self.FLOATING_MODEL = self.input_details[0]['dtype'] == np.float32

        self.INPUT_INDEX = self.input_details[0]['index']
        self.HEATMAP_INDEX = self.output_details[0]['index']
        self.OFFSET_INDEX = self.output_details[1]['index']
        self.DISPLACEMENTFWD_INDEX = self.output_details[2]['index']
        self.DISPLACEMENTBWD_INDEX = self.output_details[3]['index']

        self.OUTPUT_STRIDE = int(self.INPUT_HEIGHT / (self.output_details[0]['shape'][1] - 1))

    def get_input_dim(self):
        ''' return (INPUT_WIDTH, INPUT_HEIGHT) '''
        return self.INPUT_WIDTH, self.INPUT_HEIGHT

    def prepare_input(self, img):
        ''' img is a (height, width, 3) image. this will resize the image to the PoseNet input dimensions, and add a batch dimension. Return an image with shape (1, INPUT_HEIGHT, INPUT_WIDTH, 3). Original image is not modified. '''
        img_copy = img.copy()
        img_copy = cv2.resize(img_copy, (self.INPUT_WIDTH, self.INPUT_HEIGHT))
        img_copy = np.expand_dims(img_copy, axis=0)
        return img_copy

    def draw_keypoints_to_image(self, img, keypoints, threshold=0.5):
        ''' draw keypoints on the image img. will resize the keypoints if img size is different from INPUT_WIDTH and INPUT_HEIGHT '''
        scaleX = img.shape[1] / self.INPUT_WIDTH
        scaleY = img.shape[0] / self.INPUT_HEIGHT
        return draw_keypoints(img, keypoints, threshold=threshold, scaleX=scaleX, scaleY=scaleY)

    def predict(self, img):
        ''' invoke the TensorFlow Lite model. Return heatmaps, offsets, displacementFoward, and displacementBackward tensors '''
        img_copy = img.copy()
        if self.FLOATING_MODEL:
            img_copy = (img_copy.astype(np.float32) - 127.5) / 127.5

        self.interpreter.set_tensor(self.INPUT_INDEX, img_copy)  # load image input to INPUT_INDEX
        self.interpreter.invoke()    # run the model

        heatmaps = self.interpreter.get_tensor(self.HEATMAP_INDEX)    # obtain heatmaps
        offsets = self.interpreter.get_tensor(self.OFFSET_INDEX)      # obtain offsets
        displacementFwd = self.interpreter.get_tensor(self.DISPLACEMENTFWD_INDEX)     # obtain displacement forward
        displacementBwd = self.interpreter.get_tensor(self.DISPLACEMENTBWD_INDEX)     # obtain displacement backward

        return heatmaps, offsets, displacementFwd, displacementBwd

    def predict_singlepose(self, img):
        ''' Wrapper around decode_singlepose. Return a list of 17 keypoints '''
        img_copy = img.copy()
        heatmaps, offsets, _, _ = self.predict(img_copy)

        heatmaps = np.squeeze(heatmaps)
        offsets = np.squeeze(offsets)

        return decode_singlepose(heatmaps, offsets, self.OUTPUT_STRIDE)

def decode_singlepose(heatmaps, offsets, outputStride):
    ''' Decode heatmaps and offets output to keypoints. Return a list of keypoints, each keypoint is a tuple ((y_pos, x_pos), score) '''
    numKeypoints = heatmaps.shape[-1]

    def get_keypoint(i):
        sub_heatmap = heatmaps[:,:,i]    # heatmap corresponding to keypoint i
        y, x = np.unravel_index(np.argmax(sub_heatmap), sub_heatmap.shape)    # y, x position of the max value in heatmap
        score = sub_heatmap[y,x]    # max value in heatmap

        # convert x, y to coordinates on the input image
        y_image = y*outputStride + offsets[y, x, i]
        x_image = x*outputStride + offsets[y, x, i + numKeypoints]
        
        y_image = int(round(y_image))
        x_image = int(round(x_image))
        return (y_image, x_image), score

    keypoints = [get_keypoint(i) for i in range(numKeypoints)]
    
    return keypoints

# code from https://github.com/tensorflow/tfjs-models/tree/master/posenet
def decode_multipose(heatmaps, offsets, displacementFwd, displacementBwd, outputStride, maxPose, threshold=0.5, localMaxR=20, NmsRadius=20):
    height, width, numKeypoints = heatmaps.shape
    sqNmsRadius = NmsRadius**2
    poses = []
    # use a max heap here
    queue = []

    for keypointId in range(numKeypoints):
        sub_heatmap = heatmaps[:,:,keypointId]
        # only consider points with score >= threshold as root candidate
        candidates = np.argwhere(sub_heatmap >= threshold)

        for candidate in candidates:
            y, x = candidate
            # check if the candidate is local maximum in the local window
            # [(x-localMaxR, y-localMaxR), (x+localMaxR, y+localMaxR)]
            x0 = max(0, x-localMaxR)
            x1 = min(width, x+localMaxR+1)
            y0 = max(0, y-localMaxR)
            y1 = min(height, y+localMaxR+1)
            local_window = sub_heatmap[y0:y1, x0:x1]
    
            max_score = np.max(local_window)
            if sub_heatmap[y,x] == max_score:
                queue.append((
                    sub_heatmap[y,x],   # score
                    (y,x),              # position
                    keypointId,         # keypoint id            
                ))
    heapq._heapify_max(queue)

    # generate at most maxPose object instances
    while len(poses) < maxPose and len(queue) > 0:
        # root will be the keypoint with highest score from the queu
        root = heapq._heappop_max(queue)
        score, (y,x), keypointId = root

        # calculate position of keypoint on original input image
        # offsets has shape (width, height, 2*numKeypoints)
        y_image = y*outputStride + offsets[y, x, keypointId]
        x_image = x*outputStride + offsets[y, x, keypointId+numKeypoints]

        # reject root if it is within a disk of nmsRadius from the corresponding part of a previously detected instance
        reject = False
        for pose in poses:
            y_pose, x_pose = pose[keypointId]
            if (y_pose - y_image)**2 + (x_pose - x_image)**2 <= sqNmsRadius:
                reject = True
                break
        if reject:
            continue

        instanceKeypoints = [0] * numKeypoints
        instanceKeypoints[keypointId] = (score, (y,x))

        numEdges = len(parentToChildEdges)
        for edge in range(numEdges-1, -1, -1):  # from numEdges-1 to 0 inclusive
            sourceKeypointId = parentToChildEdges[edge]
            targetKeypointId = childToParentEdges[edge]
            if instanceKeypoints[sourceKeypointId] and not instanceKeypoints[targetKeypointId]:
                instanceKeypoints[targetKeypointId] = traverseToTargetKeypoint

        poses.append({
            keypoints: None, # get keypoints from decodePose()
            score: None, # get instance score from the instance
        })
    return queue

def traverseToTargetKeypoint(edgeId, sourceKeypoint, targetKeypointId, scores, offsets, outputStride, displacements, offsetRefineStep=2):
    y,x = sourceKeypoint
    sourceKeypointIndices = (y/outputStride, x/outputStride)
    displacement = (displacements[y,x,edgeId], displacements[y,x,edgeId])

    displacedPoint = (y+displacement[0], x+displacement[1])
    targetKeyPoint = displacedPoint
    for i in range(offsetRefineStep):
        i = 0


def draw_keypoints(img, keypoints, threshold=0.5, scaleX=1, scaleY=1):
    ''' Draw keypoints on the given image '''
    for i, keypoint in enumerate(keypoints):
        pos, score = keypoint
        if score < threshold:
            continue    # skip if score is below threshold

        # scale x and y back to original image size
        y = int(round(pos[0] * scaleY))
        x = int(round(pos[1] * scaleX))

        cv2.circle(img,(x,y),5,(0,255,0),-1)    # draw keypoint as circle
        keypoint_name = keypoint_decoder[i]
        cv2.putText(img,keypoint_name,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2) # put the name of keypoint

    return img

if __name__ == '__main__':
    # set up posenet
    model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    posenet = PoseNet(model_path)

    # read image and prepare input to shape (1, height, width, 3)
    img = cv2.imread('7535.jpg')
    img = cv2.resize(img, (1920, 1080))
    img_input = posenet.prepare_input(img)

    # apply model
    keypoints = posenet.predict_singlepose(img_input)
    # draw keypoints on original image
    posenet.draw_keypoints_to_image(img, keypoints)

    cv2.imshow('round=True', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()