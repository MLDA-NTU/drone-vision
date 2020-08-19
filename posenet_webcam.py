import tensorflow as tf
import numpy as np
import cv2

def parse_output(heatmap_data,offset_data, threshold):
    '''
    Input:
        heatmap_data - hetmaps for an image. Three dimension array
        offset_data - offset vectors for an image. Three dimension array
        threshold - probability threshold for the keypoints. Scalar value
    Output:
        array with coordinates of the keypoints and flags for those that have
        low probability
    '''

    joint_num = heatmap_data.shape[-1]
    pose_kps = np.zeros((joint_num,3), np.uint32)

    for i in range(heatmap_data.shape[-1]):

        joint_heatmap = heatmap_data[...,i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos/8*257,dtype=np.int32)
        pose_kps[i,0] = int(remap_pos[0] + offset_data[max_val_pos[0],max_val_pos[1],i])
        pose_kps[i,1] = int(remap_pos[1] + offset_data[max_val_pos[0],max_val_pos[1],i+joint_num])
        max_prob = np.max(joint_heatmap)

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

model_path = "posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = input_details[0]['dtype'] == np.float32


cap = cv2.VideoCapture(0)

# Set VideoCaptureProperties
# Width and Height
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    img_input = cv2.resize(img.copy(), (width, height))
    img_input = np.expand_dims(img_input, axis=0)

    if floating_model:
        img_input = (np.float32(img_input) - 127.5) / 127.5

    # Process template image
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    img_output_data = interpreter.get_tensor(output_details[0]['index'])
    img_offset_data = interpreter.get_tensor(output_details[1]['index'])
    img_heatmaps = np.squeeze(img_output_data)
    img_offsets = np.squeeze(img_offset_data)

    img_show = np.squeeze((img_input.copy()*127.5+127.5)/255.0)
    img_show = np.array(img_show*255,np.uint8)
    img_kps = parse_output(img_heatmaps,img_offsets,0.3)
    cv2.imshow("pose", draw_kps(img_show.copy(),img_kps))

    # cv2.imshow("Video from webcam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()