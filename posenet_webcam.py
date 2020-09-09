import tensorflow as tf
import numpy as np
import cv2

from posenet import PoseNet, detect_pose

# itialize posenet from the package
model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
posenet = PoseNet(model_path)

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
    img_input = posenet.prepare_input(img)

    # get keypoints for single pose estimation. it is a list of 17 keypoints
    keypoints = posenet.predict_singlepose(img_input)

    # draw keypoints to the original image
    threshold = 0.0
    posenet.draw_pose(img, keypoints, threshold=threshold)
    poses = detect_pose(keypoints)
    detected_poses = [pose for pose, detected in poses.items() if detected]
    detected_poses = ' '.join(detected_poses) if detected_poses else 'None'
    
    cv2.putText(img, f'{detected_poses} detected', (0,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("pose", img)                 # show the image with keypoints
    if cv2.waitKey(1) & 0xFF == ord('q'):   # terminate window when press q
        break

cv2.destroyAllWindows()