import numpy as np
from openvino.inference_engine import IENetwork, IECore
import cv2
import paddleseg.transforms as T


def get_net(model_xml, model_bin, device_name="MYRIAD"):
    ie = IECore()
    # Read IR
    net = IENetwork(model=model_xml, weights=model_bin)
    input_blob = next(iter(net.inputs))
    exec_net = ie.load_network(network=net, device_name=device_name)
    del net
    return exec_net, input_blob

def show_img(img, channel_first):
    visual = img
    if channel_first:
        visual = img.transpose(1,2,0)
    img = cv2.resize(visual, (600,600))
    cv2.imshow('segmentation', img)
    
def save_img(img, img_fn):
    cv2.imwrite(img_fn, img)

model_xml = r'/media/winstonfan/Workspace/Work/MyBuddy/models/paddle/static/openvino/UINT8/road_seg_half.xml'
model_bin = r'/media/winstonfan/Workspace/Work/MyBuddy/models/paddle/static/openvino/UINT8/road_seg_half.bin'

transforms = [
            T.Resize(target_size=(224,224))
        ]

# Run inference
img_fn = '/media/winstonfan/Workspace/Work/MyBuddy/Data/videos/boxhill/boxhill_079.jpeg'

img = cv2.imread(img_fn)

img, _ = T.Compose(transforms)(img)
# add an new axis in front
img_input = img[np.newaxis, :]
exec_net, input_blob = get_net(model_xml, model_bin)
result = exec_net.infer(inputs={input_blob: img_input})
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
save_img(img_overlayed, 'demo.jpg')
