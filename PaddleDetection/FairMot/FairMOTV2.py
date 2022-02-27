from collections import defaultdict, deque
import cv2
import numpy as np
from openvino.inference_engine import IECore
import paddleseg.transforms as T

from ppdet.modeling.mot.tracker import JDETracker

def get_net():
    ie = IECore()
    net = ie.read_network(model="/media/winstonfan/Workspace/Work/Intel/Models/Paddle/Tracking/PPTrackingMOT/selfexported/fairmot_hrnetv2_w18_dlafpn_30e_576x320/pptrackingfairmot320_576_op12.onnx")    
    exec_net = ie.load_network(network=net, device_name="CPU")
    return net, exec_net

def get_output_names(net):
    output_names = [key for key in net.outputs]
    # 0:'translated_layer/scale_0.tmp_0', 1:'translated_layer/scale_1.tmp_0'
    return output_names

def prepare_input():
    # Read IR
    # ['im_shape', 'image', 'scale_factor']
    # input_names = [key for key in net.inputs]
    transforms = [T.Resize(target_size=(576, 320)), T.Normalize()]

    img_file = "/media/winstonfan/Workspace/Work/Intel/Code/Intel-OpenVINO-Paddle/images/street.jpeg"
    img = cv2.imread(img_file)
    normalized_img, _ = T.Compose(transforms)(img)

    # add an new axis in front
    img_input = normalized_img[np.newaxis, :]
    input = {"image": img_input, "im_shape": [320, 576], "scale_factor": [0.234, 0.281]}
    return input, img

def predict(exec_net, input):
    result = exec_net.infer(input)
    return result


video = "/media/winstonfan/Workspace/Work/MyBuddy/Data/videos/runner.mp4"
# res = ppdet.engine.Tracker.mot_predict(video_file=video,frame_rate=30,output_dir="./")

net, exec_net = get_net()
output_names = get_output_names(net)
del net
input, img = prepare_input()
result = predict(exec_net, input)



def postprocess(pred_dets, pred_embs, threshold):
    tracker = JDETracker()
    tracker.update(pred_dets, pred_embs)
    online_targets_dict = tracker.update(pred_dets, pred_embs)

    online_tlwhs = defaultdict(list)
    online_scores = defaultdict(list)
    online_ids = defaultdict(list)
    for cls_id in range(1):
        online_targets = online_targets_dict[cls_id]
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            tscore = t.score
            if tscore < threshold: continue
            if tlwh[2] * tlwh[3] <= tracker.min_box_area:
                continue
            if tracker.vertical_ratio > 0 and tlwh[2] / tlwh[3] > tracker.vertical_ratio:
                continue
            online_tlwhs[cls_id].append(tlwh)
            online_ids[cls_id].append(tid)
            online_scores[cls_id].append(tscore)
    return online_tlwhs, online_scores, online_ids

# postprocess
online_tlwhs, online_scores, online_ids = postprocess(pred_dets=result[output_names[0]], pred_embs=result[output_names[1]], threshold=0.5)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot_tracking_dict(image,
                       num_classes,
                       tlwhs_dict,
                       obj_ids_dict,
                       scores_dict,
                       frame_id=0,
                       fps=0.,
                       ids2names=[],
                       do_entrance_counting=False,
                       entrance=None,
                       records=None,
                       center_traj=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    text_scale = max(1, image.shape[1] / 1600.)
    text_thickness = 2
    line_thickness = max(1, int(image.shape[1] / 500.))

    if num_classes == 1:
        if records is not None:
            start = records[-1].find('Total')
            end = records[-1].find('In')
            cv2.putText(
                im,
                records[-1][start:end], (0, int(40 * text_scale)),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale, (0, 0, 255),
                thickness=2)

    if num_classes == 1 and do_entrance_counting:
        entrance_line = tuple(map(int, entrance))
        cv2.rectangle(
            im,
            entrance_line[0:2],
            entrance_line[2:4],
            color=(0, 255, 255),
            thickness=line_thickness)
        # find start location for entrance counting data
        start = records[-1].find('In')
        cv2.putText(
            im,
            records[-1][start:-1], (0, int(60 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale, (0, 0, 255),
            thickness=2)

    for cls_id in range(num_classes):
        tlwhs = tlwhs_dict[cls_id]
        obj_ids = obj_ids_dict[cls_id]
        scores = scores_dict[cls_id]
        cv2.putText(
            im,
            'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale, (0, 0, 255),
            thickness=2)

        record_id = set()
        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            center = tuple(map(int, (x1 + w / 2., y1 + h / 2.)))
            obj_id = int(obj_ids[i])
            if center_traj is not None:
                record_id.add(obj_id)
                if obj_id not in center_traj[cls_id]:
                    center_traj[cls_id][obj_id] = deque(maxlen=30)
                center_traj[cls_id][obj_id].append(center)

            id_text = '{}'.format(int(obj_id))
            if ids2names != []:
                id_text = '{}_{}'.format(ids2names[cls_id], id_text)
            else:
                id_text = 'class{}_{}'.format(cls_id, id_text)

            _line_thickness = 1 if obj_id <= 0 else line_thickness
            color = get_color(abs(obj_id))
            cv2.rectangle(
                im,
                intbox[0:2],
                intbox[2:4],
                color=color,
                thickness=line_thickness)
            cv2.putText(
                im,
                id_text, (intbox[0], intbox[1] - 10),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale, (0, 0, 255),
                thickness=text_thickness)

            if scores is not None:
                text = '{:.2f}'.format(float(scores[i]))
                cv2.putText(
                    im,
                    text, (intbox[0], intbox[1] + 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale, (0, 255, 255),
                    thickness=text_thickness)
        if center_traj is not None:
            for traj in center_traj:
                for i in traj.keys():
                    if i not in record_id:
                        continue
                    for point in traj[i]:
                        cv2.circle(im, point, 3, (0, 0, 255), -1)
    return im

online_im = plot_tracking_dict(
    img,
    1,
    online_tlwhs,
    online_ids,
    online_scores,
    frame_id=0)

cv2.imshow("wow", online_im)
# mot predict
# mot_jde_infer.predict_naive(model_dir, video_file, image_dir, device, threshold, output_dir)

# result[ouput_names[1]].shape
# (500, 128)
# result[ouput_names[0]].shape
# (500, 6)