import numpy as np
import cv2

keypoint_decoder = [
    "nose",
    "leftEye",
    "rightEye",
    "leftEar",
    "rightEar",
    "leftShoulder",
    "rightShoulder",
    "leftElbow",
    "rightElbow",
    "leftWrist",
    "rightWrist",
    "leftHip",
    "rightHip",
    "leftKnee",
    "rightKnee",
    "leftAnkle",
    "rightAnkle",
]

# Pairs represents the lines connected from joints
# e.g. (5,6) is from leftShoulder to rightShoulder
# https://www.tensorflow.org/lite/models/pose_estimation/overview
parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]

# code from https://github.com/tensorflow/tfjs-models/tree/master/posenet
def decode_multipose(heatmaps, offsets, maxPose, localMaxR, outputStride, threshold=0.5):
    height, width, numKeypoints = heatmaps.shape
    poses = []
    queue = []
    for keypointId in range(numKeypoints):
        # only consider points with score >= threshold as root candidate
        candidates = np.argwhere(heatmaps[:,:,keypointId] >= threshold)

        for candidate in candidates:
            x, y = candidate
            # check if the candidate is local maximum in the local window
            # [(x-localMaxR, y-localMaxR), (x+localMaxR, y+localMaxR)]
            x0 = max(0, x-localMaxR)
            x1 = min(width, x+localMaxR+1)
            y0 = max(0, y-localMaxR)
            y1 = min(height, y+localMaxR+1)
            subarray = heatmaps[x0:x1, y0:y1, keypointId]
    
            max_idx = np.argmax(subarray)
            xmax, ymax = np.unravel_index(max_idx, subarray.shape)
            if xmax == x and ymax == y:
                queue.append({
                    'score': heatmaps[x,y,keypointId],
                    'pos': (x,y,keypointId)
                })

    # generate at most maxPose object instances
    while len(poses) < maxPose and len(queue) > 0:
        root = queue.pop()
        x, y, keypointId = root.pos

        # offsets has shape (width, height, 2*numKeypoints)
        # first numKeypoints is x, second numKeypoints is y
        x_original = x*outputStride + offsets[x, y, keypointId]
        y_original = y*outputStride + offsets[x, y, keypointId+numKeypoints]

        # reject root if it is within a disk of nmsRadius from the corresponding part of a previously detected instance

        poses.append({
            keypoints: None, # get keypoints from decodePose()
            score: None, # get instance score from the instance
        })
    return queue

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
    scores = np.random.randn(64, 64, 10)
    print(parse_output_multipose(scores, None, 10))