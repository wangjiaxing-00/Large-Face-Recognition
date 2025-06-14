from __future__ import division
import numpy as np
import os.path as osp
import cv2

def check_face_num(face_detection_result_dict):
    num = face_detection_result_dict['bboxes'].shape[0]
    return num

def select_one_biggest_face(detection_dict,face_num):
    bboxes = detection_dict['bboxes']
    kpss = detection_dict['kpss']
    if face_num>1:
        area_list = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1, y1, x2, y2, score = bbox.astype(int)
            area = abs(x2-x1)*abs(y2-y1)
            area_list.append(area)
        import numpy
        area_array = numpy.array(area_list)
        big_area_index = area_array.argmax()
        temp_bboxes = bboxes[big_area_index]
        temp_kpss = kpss[big_area_index]
        detection_dict['bboxes'] = temp_bboxes
        detection_dict['kpss'] = temp_kpss
        return detection_dict

font = cv2.FONT_HERSHEY_SIMPLEX

def show_detect_result(img,detection_dict):
    bboxes = detection_dict['bboxes']
    kpss = detection_dict['kpss']
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if kpss is not None:
            kps = kpss[i]
            for num,kp in enumerate(kps):
                kp = kp.astype(int)
                cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
                cv2.putText(img, str(num),tuple(kp), font,1, (0, 255, 0), 2)
    cv2.imshow('img_show',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

class SD:
    def __init__(self, model_file=None, session=None):
        import onnxruntime
        self.model_file = model_file
        self.session = session
        self.taskname = 'detection'
        self.batched = False
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(self.model_file, providers=['CUDAExecutionProvider'])
        self.center_cache = {}
        self.nms_thresh = 0.4
        self._init_vars()

    def _init_vars(self):
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = self.session.get_outputs()
        if len(outputs[0].shape) == 3:
            self.batched = True
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names
        self.use_kps = False
        self._num_anchors = 1
        if len(outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

    def prepare(self, ctx_id, **kwargs):
        nms_thresh = kwargs.get('nms_thresh', None)
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        input_size = kwargs.get('input_size', None)
        if input_size is not None:
            if self.input_size is not None:
                print('warning: det_size is already set in sd model, ignore')
            else:
                self.input_size = input_size

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + fmc][0]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2][0] * stride
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + fmc]
                bbox_preds = bbox_preds * stride
                if self.use_kps:
                    kps_preds = net_outs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers
            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.5, input_size=None, max_num=0, metric='default'):
        assert input_size is not None or self.input_size is not None
        input_size = self.input_size if input_size is None else input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] -
                                              det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(
                values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

def check_face_horzion(detection_result):
    bbox = detection_result['bboxes']
    x1, y1, x2, y2 = bbox[0:4].astype(int)
    height = abs(y1-y2)
    width = abs(x1-x2)
    rate_h_w = height/width
    if rate_h_w < 1.0:
        return True
    else:
        return False

def face_detect_model(img,detector,visualize=False):
    bboxes, kpss = detector.detect(img, 0.5, input_size=(320, 320))
    result_dict = {'bboxes':bboxes,'kpss':kpss}
    if visualize:
        show_detect_result(img, result_dict)
    face_num = check_face_num(result_dict)
    face_detection_result_dict,has_result_flag = select_one_biggest_face(result_dict, face_num)
    if has_result_flag:
        horizon_flag = check_face_horzion(face_detection_result_dict)
        if horizon_flag:
            return face_detection_result_dict, horizon_flag
        else:
            return face_detection_result_dict, horizon_flag
    else:
        return face_detection_result_dict, None

def face_detect_model_pure(img, detector):
    ret = detector.detect(img, 0.5, input_size=(320, 320))
    bboxes, kpss = ret
    result_dict = {'bboxes': bboxes, 'kpss': kpss}
    face_num = check_face_num(result_dict)
    if face_num>1:
        face_detection_result_dict = select_one_biggest_face(result_dict, face_num)
        # 人脸检测测试可视化
        # x1, y1, x2, y2 = face_detection_result_dict['bboxes'][0:4].astype(np.int)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.imshow('img_show', img)
        # cv2.imwrite('0.jpg', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return face_detection_result_dict
    elif face_num == 1:
        face_detection_result_dict = {'bboxes': bboxes[0], 'kpss': kpss[0]}
        #  人脸检测测试可视化
        # x1, y1, x2, y2= face_detection_result_dict['bboxes'][0:4].astype(np.int)
        # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # cv2.imshow('img_show', img)
        # cv2.imwrite('xuan.jpg', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return face_detection_result_dict
    elif face_num == 0:
        face_detection_result_dict = None
    return face_detection_result_dict



