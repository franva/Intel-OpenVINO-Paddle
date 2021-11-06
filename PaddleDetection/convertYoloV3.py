from build.lib import paddle2onnx
import paddle

model_prefix = "/Paddle/Detection/yolov3_mobilenet_v3_large_270e_voc/model"
model = paddle.jit.load(model_prefix)
input_shape_dict = {
    "image": [1, 3, 608, 608],
    "scale_factor": [1, 2],
    "im_shape": [1, 2]
    }
onnx_model = paddle2onnx.run_convert(model, input_shape_dict=input_shape_dict, opset_version=11)

with open("./yolov3.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
