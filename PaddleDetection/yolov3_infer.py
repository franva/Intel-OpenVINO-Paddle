from experimental.openvino_ppdet.yolov3_infer import YOLOv3

xml_file = "/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Detection/yolov3_mobilenet_v3_large_270e_voc_origin/OpenVINO/yolov3.xml"
bin_file = "/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Detection/yolov3_mobilenet_v3_large_270e_voc_origin/OpenVINO/yolov3.bin"
model = YOLOv3(xml_file=xml_file,
               bin_file=bin_file,
               model_input_shape=[608, 608])
# boxes = model.predict("/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Detection/street.jpeg", 
# visualize_out="/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Detection/result.jpg", threshold=0.1)
boxes = model.predict("/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Detection/street.jpeg", threshold=0.1)

for box in boxes:
    print(box)
