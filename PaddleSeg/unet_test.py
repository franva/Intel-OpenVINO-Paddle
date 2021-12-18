import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import paddleseg.transforms as T

# Please update the pathes to xml and bin files respectively on your machine
model_xml = r'/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Segmentation/unet/OV/unet.xml'
model_bin = r'/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Segmentation/unet/OV/unet.bin'


ie = IECore()

# Read IR
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name="CPU")

del net

transforms = [
            T.Resize(target_size=(128, 128)),
            T.Normalize()
        ]

img_file = r'/media/winstonfan/Workspace/temp/Drishti-GS1_files/Drishti-GS1_files/Test/Images/drishtiGS_087.png'
mark_colors = np.array([[0,0,0],[0,255,0]])

img = cv2.imread(img_file)  # (1835, 2049, 3)
normalized_img, _ = T.Compose(transforms)(img) # (3, 128, 128)

# add an new axis in front
img_input = normalized_img[np.newaxis, :]
result = exec_net.infer(inputs={input_blob: img_input})

img_segmentation = result['save_infer_model/scale_0.tmp_1']
img_segmentation = np.squeeze(img_segmentation) # (128, 128)

img_mask = mark_colors[img_segmentation].astype('uint8') # (1835, 2049, 3)
cv2.imwrite('unet_eye_marked_128.png', img_mask)

# scale back to the original size
img_segmentation = cv2.resize(img_segmentation, dsize= img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR_EXACT)
img_mask = mark_colors[img_segmentation].astype('uint8') # (1835, 2049, 3)
img_overlayed = cv2.addWeighted(img, 1, img_mask, 1, 0.5)

cv2.imwrite("unet_eye_marked.png", img_overlayed)
