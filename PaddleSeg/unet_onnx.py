import cv2
import numpy as np
import paddleseg.transforms as T
import onnxruntime as rt
import time as t

path = "PaddleSeg/models/unetv2.onnx"
# session = rt.InferenceSession(path, providers=['OpenVINOExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
# session = rt.InferenceSession(path, providers=['OpenVINOExecutionProvider'])
# session = rt.InferenceSession(path, providers=['CUDAExecutionProvider'])
session = rt.InferenceSession(path, providers=['CPUExecutionProvider'])
transforms = [T.Resize(target_size=(128, 128)), T.Normalize()]

img_file = r'PaddleSeg/drishtiGS_087.png'
img = cv2.imread(img_file).astype('float32')
normalized_img, _ = T.Compose(transforms)(img)
img_input = normalized_img[np.newaxis, :]

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

t1 = t.time()
result = session.run([output_name], {input_name: img_input})
t2 = t.time()
print(f'Time consumed: {t2-t1}')


mark_colors = np.array([[0,0,0],[0,255,0]])
img_segmentation = np.squeeze(result) # (128, 128)

img_mask = mark_colors[img_segmentation].astype('uint8') # (1835, 2049, 3)
cv2.imwrite('unet_eye_marked_128_onnx.png', img_mask)
cv2.imshow("mask", img_mask)

# scale back to the original size
img_segmentation = cv2.resize(np.uint8(img_segmentation), dsize= img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR_EXACT)
img_mask = mark_colors[img_segmentation].astype('uint8') # (1835, 2049, 3)
img_overlayed = cv2.addWeighted(np.uint8(img), 1, img_mask, 1, 0.5)

cv2.imwrite("unet_eye_marked_onnx.png", img_overlayed)

img_overlayed = cv2.resize(img_overlayed, dsize=(670, 600))
cv2.imshow("img with mask", img_overlayed)
cv2.waitKey(0)