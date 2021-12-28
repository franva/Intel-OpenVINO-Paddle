import cv2
import numpy as np
from scipy.special import softmax
import picodet_classes
from picodet_classes import COLORS, CLASS_NAMES

strides=[8, 16, 32]
reg_max=7
input_shape=[320, 320]
num_candidate=1000
prob_threshold = 0.4
iou_threshold=0.3,
top_k=-1

def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(
                current_box, axis=0), )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def warp_boxes(boxes, M, width, height):
    """Apply transform to boxes
    Copy from picodet/data/transform/warp.py
    """
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """
    Get resize matrix for resizing raw img to input size
    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @Rs @C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs

def overlay_bbox_cv(img, all_box, class_names):
    """Draw result boxes
    Copy from picodet/util/visualization.py
    """
    all_box.sort(key=lambda v: v[5])
    for box in all_box:
        label, x0, y0, x1, y1, score = box
        color = (picodet_classes.COLORS[label] * 255).astype(np.uint8).tolist()
        text = "{}:{:.1f}%".format(class_names[label], score * 100)
        txt_color = (0, 0, 0) if np.mean(picodet_classes.COLORS[label]) > 0.5 else (255, 255,
                                                                     255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.5, 2)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 - txt_size[1] - 1),
            (x0 + txt_size[0] + txt_size[1], y0 - 1),
            color,
            -1, )
        cv2.putText(img, text, (x0, y0 - 1), font, 0.5, txt_color, thickness=1)
    return img

def draw_box(raw_img, bbox, label, score):
    img = raw_img.copy()
    all_box = [[x, ] + y + [z, ]
                   for x, y, z in zip(label, bbox.tolist(), score)]
    img_draw = overlay_bbox_cv(img, all_box, picodet_classes.CLASS_NAMES)
    return img_draw


def postprocess(scores, raw_boxes, raw_shape):
        ResizeM = get_resize_matrix((raw_shape[1], raw_shape[0]), input_shape, True)
        # generate centers
        decode_boxes = []
        select_scores = []
        for stride, box_distribute, score in zip(strides, raw_boxes,
                                                 scores):
            # centers
            fm_h = input_shape[0] / stride
            fm_w = input_shape[1] / stride
            h_range = np.arange(fm_h)
            w_range = np.arange(fm_w)
            ww, hh = np.meshgrid(w_range, h_range)
            ct_row = (hh.flatten() + 0.5) * stride
            ct_col = (ww.flatten() + 0.5) * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # box distribution to distance
            reg_range = np.arange(reg_max + 1)
            box_distance = box_distribute.reshape((-1, reg_max + 1))
            box_distance = softmax(box_distance, axis=1)
            box_distance = box_distance * np.expand_dims(reg_range, axis=0)
            box_distance = np.sum(box_distance, axis=1).reshape((-1, 4))
            box_distance = box_distance * stride

            # top K candidate
            topk_idx = np.argsort(score.max(axis=1))[::-1]
            topk_idx = topk_idx[:num_candidate]
            center = center[topk_idx]
            score = score[topk_idx]
            box_distance = box_distance[topk_idx]

            # decode box
            decode_box = center + [-1, -1, 1, 1] * box_distance

            select_scores.append(score)
            decode_boxes.append(decode_box)

        # nms
        bboxes = np.concatenate(decode_boxes, axis=0)
        confidences = np.concatenate(select_scores, axis=0)
        picked_box_probs = []
        picked_labels = []
        for class_index in range(0, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = bboxes[mask, :]
            box_probs = np.concatenate(
                [subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(
                box_probs,
                iou_threshold=iou_threshold,
                top_k=top_k, )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)

        # resize output boxes
        picked_box_probs[:, :4] = warp_boxes(picked_box_probs[:, :4],
                                             np.linalg.inv(ResizeM),
                                             raw_shape[1], raw_shape[0])
        return (
            picked_box_probs[:, :4].astype(np.int32),
            np.array(picked_labels),
            picked_box_probs[:, 4], )

