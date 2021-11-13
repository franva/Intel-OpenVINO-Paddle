# Notice: this file is meant to be used within Paddle2ONNX Github repo.
from experimental.openvino_ppdet.yolov3_infer import YOLOv3

xml_file = "./OpenVINOModels/yolov3_320_V3.xml"
bin_file = "./OpenVINOModels/yolov3_320_V3.bin"
model = YOLOv3(xml_file=xml_file,
               bin_file=bin_file,
               model_input_shape=[320, 320])
boxes = model.predict("<your photo>", visualize_out="result.jpg", threshold=0.1)

for box in boxes:
    print(box)
