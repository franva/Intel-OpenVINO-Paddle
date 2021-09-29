import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import paddleseg.transforms as T

from colors_pascalvoc import ColorMap_PASCALVOC

# Please update the pathes to xml and bin files respectively on your machine
model_xml = r'/Models/Paddle/Segmentation/DeeplabV3Plus/OpenVINO/pascalvoc.xml'
model_bin = r'/Models/Paddle/Segmentation/DeeplabV3Plus/OpenVINO/pascalvoc.bin'


ie = IECore()

# Read IR
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name="CPU")

del net

transforms = [
            T.Resize(target_size=(512,512)),
            T.Normalize()
        ]

def show_img(img, window_name, channel_first=True):
    visual = img
    if channel_first:
        visual = img.transpose(1,2,0)
        visual = cv2.resize(visual, (600, 600))
    else:
        visual = cv2.resize(visual, (600,600))
    cv2.imshow(window_name, visual)
    
def save_img(img, img_fn):
    cv2.imwrite(img_fn, img)


# Run inference
video = '/Testing Videos/mel_highway.mp4'

cap = cv2.VideoCapture(video)
read_successfully, frame = cap.read()

while read_successfully:    

    if read_successfully == False:
        continue

    resized_frame, tesrt = T.Compose([transforms[0]])(frame)
    img, _ = T.Compose(transforms)(frame)

    # add an new axis in front
    img_input = img[np.newaxis, :]
    result = exec_net.infer(inputs={input_blob: img_input})

    img_segmentation = result['save_infer_model/scale_0.tmp_1']
    img_segmentation = np.squeeze(img_segmentation)
    class_colors = ColorMap_PASCALVOC.COLORS
    class_colors = np.asarray(class_colors, dtype=np.uint8)
    img_mask = class_colors[img_segmentation]

    img_mask = img_mask.transpose(2, 0, 1)

    img_overlayed = cv2.addWeighted(resized_frame, 1, img_mask, 1, 0.5)
    img_overlayed = img_overlayed.transpose(1,2,0)
    img_overlayed = cv2.cvtColor(img_overlayed, cv2.COLOR_RGB2BGR)
    show_img(img_overlayed, 'overlayed', False)
    show_img(img_mask, 'mask', True)

    if cv2.waitKey(1) == ord('q'):
        break

    read_successfully, frame = cap.read()

cap.release()
cv2.destroyAllWindows()