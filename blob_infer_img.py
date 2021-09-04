import time
import cv2
import numpy as np
from openvino.inference_engine import IECore
import paddleseg.transforms as T

model_blob = r'/openvino/blob/model_fp16.blob'

def get_net(model_blob, device_name='MYRIAD'):
    ie = IECore()
    exec_net = ie.import_network(model_blob, device_name = device_name)
    input_blob = next(iter(exec_net.inputs))
    return exec_net, input_blob

def save_img(img, img_fn):
    cv2.imwrite(img_fn, img)

# Run inference
img_fn = '/data/boxhill_079.jpeg'
img = cv2.imread(img_fn)

transforms = [
            T.Resize(target_size=(224,224))
        ]

img, _ = T.Compose(transforms)(img)
# add an new axis in front
img_input = img[np.newaxis, :]
t1 = time.time()
exec_net, input_blob = get_net(model_blob)
result = exec_net.infer(inputs={input_blob: img_input})
print(' time used : {}'.format(time.time() - t1))
img_segmentation = result['save_infer_model/scale_0.tmp_1']
# img_segmentation is int32
img_segmentation = np.squeeze(img_segmentation)
class_colors = [[0,0,0], [0,255,0]]
class_colors = np.asarray(class_colors, dtype=np.uint8)
img_mask = class_colors[img_segmentation]
img_mask, _ = T.Compose(transforms)(img_mask)
img_overlayed = cv2.addWeighted(img, 0.8, img_mask, 0.2, 0.5)
img_overlayed = img_overlayed.transpose(1,2,0)
img_overlayed = cv2.cvtColor(img_overlayed, cv2.COLOR_RGB2BGR)
save_img(img_overlayed, "demo2.jpg")
