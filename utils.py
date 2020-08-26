import numpy as np

# code from https://github.com/tensorflow/tfjs-models/tree/master/posenet
def parse_output_multipose(scores, offsets, maxPose, localMaxR, outputStride, threshold=0.5):
    height, width, numKeypoints = scores.shape
    poses = []
    queue = []
    for keypointId in range(numKeypoints):
        # only consider points with score >= threshold as root candidate
        candidates = np.argwhere(scores[:,:,keypointId] >= threshold)

        for candidate in candidates:
            x, y = candidate
            # check if the candidate is local maximum in the local window
            # [(x-localMaxR, y-localMaxR), (x+localMaxR, y+localMaxR)]
            x0 = max(0, x-localMaxR)
            x1 = min(width, x+localMaxR+1)
            y0 = max(0, y-localMaxR)
            y1 = min(height, y+localMaxR+1)
            subarray = scores[x0:x1, y0:y1, keypointId]
    
            max_idx = np.argmax(subarray)
            xmax, ymax = np.unravel_index(max_idx, subarray.shape)
            if xmax == x and ymax == y:
                queue.append({
                    'score': scores[x,y,keypointId],
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

if __name__ == '__main__':
    scores = np.random.randn(64, 64, 10)
    print(parse_output_multipose(scores, None, 10))