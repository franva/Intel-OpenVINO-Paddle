import paddle
import paddle2onnx

# /media/winstonfan/Workspace/Work/Intel/Models/Paddle/Tracking/PPTrackingMOT/fairmot_hrnetv2_w18_dlafpn_30e_576x320
base_path = "/fairmot_hrnetv2_w18_dlafpn_30e_576x320"
model_prefix = base_path + "/model"
model = paddle.jit.load(model_prefix)
input_shape_dict = {
    "image": [1, 3, 320, 576],
    "scale_factor": [1, 2],
    "im_shape": [1, 2]
    }
onnx_model = paddle2onnx.run_convert(model, input_shape_dict=input_shape_dict,  opset_version=12)

with open(base_path + "/pptrackingfairmot320_576_op12_v5.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

