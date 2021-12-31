from tqdm import tqdm
from pathlib import Path
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore
import paddleseg.transforms as T

from tools import draw_box, postprocess

# Please update the pathes to xml and bin files respectively on your machine
path = Path(__file__).parent
model_xml = str(path / r'model/picodet_s_320_coco.xml')
model_bin = str(path / r'model/picodet_s_320_coco.bin')


ie = IECore()

# Read IR
net = IENetwork(model=model_xml, weights=model_bin)
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
exec_net = ie.load_network(network=net, device_name="CPU")

del net

transforms = [T.Resize(target_size=(320, 320)),
              T.Normalize()]


score_out_name = ["save_infer_model/scale_0.tmp_1", "save_infer_model/scale_1.tmp_1",
        "save_infer_model/scale_2.tmp_1", "save_infer_model/scale_3.tmp_1"]

boxes_out_name = ["save_infer_model/scale_4.tmp_1", "save_infer_model/scale_5.tmp_1",
                "save_infer_model/scale_6.tmp_1", "save_infer_model/scale_7.tmp_1"]

def squeeze_results(infer_result, score_out_name, boxes_out_name):
    scores = [np.squeeze(infer_result[x]) for x in score_out_name]
    boxes = [np.squeeze(infer_result[x]) for x in boxes_out_name]
    return scores, boxes

def detect(img_file):

    if (isinstance(img_file, str)):
        img = cv2.imread(img_file)
    else:
        img = img_file
    
    raw_shape = img.shape
    normalized_img, _ = T.Compose(transforms)(img)

    # add an new axis in front
    img_input = normalized_img[np.newaxis, :]
    result = exec_net.infer(inputs={input_blob: img_input})
    scores, boxes = squeeze_results(result, score_out_name, boxes_out_name)
    return postprocess(scores, boxes, raw_shape)  # bbox, label, score

def detect_folder(img_fold, result_path):
    img_fold = Path(img_fold)
    result_path = Path(result_path)
    result_path.mkdir(parents=True, exist_ok=True)

    img_name_list = filter(
        lambda x: str(x).endswith(".png") or str(x).endswith(".jpg"), img_fold.iterdir())
    img_name_list = list(img_name_list)

    print(f"find {len(img_name_list)} images")

    for img_path in tqdm(img_name_list):
        img = cv2.imread(str(img_path))
        detect_result = detect(img)
        if detect_result is None:
            continue
        bbox, label, score = detect_result
        img_draw = draw_box(img, bbox, label, score)
        save_path = str(result_path / img_path.name.replace(".png", ".jpg"))
        cv2.imwrite(save_path, img_draw)

folder_path = '<path to the images that you want to test>'
detect_folder(folder_path, folder_path + '/results')