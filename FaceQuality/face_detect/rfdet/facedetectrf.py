import time
import numpy as np
import torch
import torch.nn as nn
from face_detect.rfdet.rfnet import FaceDetectRFNet
from utils.anchors import Anchors
from utils.config import cfg_mnet, cfg_re50
from utils.utils import letterbox_image, preprocess_input
from utils.utils_bbox import (decode, decode_landm, non_max_suppression,
                              facedetectrf_correct_boxes)
import os

current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FaceDetectRF(object):
    # /home/ai/project/Face/face1vs1_source/face_detect_model/rf/facedetect_rf_r.pth
    _defaults = {
        # 修改成相对路径
        "model_path": os.path.join(current_path, 'face_detect_model/rf', 'Retinaface_resnet50.pth'),
        "backbone": 'rn',
        "confidence": 0.5,
        "nms_iou": 0.45,
        "input_shape": [256, 256, 3],
        "letterbox_image": True,
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if self.backbone == "mb":
            self.cfg = cfg_mnet
        else:
            self.cfg = cfg_re50
        if self.letterbox_image:
            self.anchors = Anchors(self.cfg, image_size=[self.input_shape[0], self.input_shape[1]]).get_anchors()
        self.generate()

    def generate(self):
        self.net = FaceDetectRFNet(cfg=self.cfg, mode='eval').eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def detect_image(self, image):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        total_boxes = []
        points = []
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        with torch.no_grad():
            if self.cuda:
                image = torch.from_numpy(preprocess_input(image)).type(torch.FloatTensor).cuda(device='cuda:0')
                image = image.permute(2,0,1).unsqueeze(0)
                self.anchors = self.anchors.cuda()
            else:
                image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms) <= 0:
                total_boxes = np.array(total_boxes)
                points = np.array(points)
                return total_boxes, points
            if self.letterbox_image:
                boxes_conf_landms = facedetectrf_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.input_shape[0], self.input_shape[1]]),
                                                             np.array([im_height, im_width]))
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
        for b in boxes_conf_landms:
            b_box = []
            point = []
            for b_i in range(5):
                b_box.append(b[b_i])
            for p_i in range(10):
                point.append(b[5 + p_i])
            total_boxes.append(b_box)
            points.append(point)
        total_boxes = np.array(total_boxes)
        points = np.array(points)
        return total_boxes,points

    def get_FPS(self, image, test_interval):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()
            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                loc, conf, landms = self.net(image)
                boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
                conf = conf.data.squeeze(0)[:, 1:2]
                landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
                boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
                boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image):
        image = np.array(image, np.float32)
        im_height, im_width, _ = np.shape(image)
        total_boxes = []
        points = []
        scale = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0]
        ]
        scale_for_landmarks = [
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0], np.shape(image)[1], np.shape(image)[0],
            np.shape(image)[1], np.shape(image)[0]
        ]
        if self.letterbox_image:
            image = letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
        else:
            self.anchors = Anchors(self.cfg, image_size=(im_height, im_width)).get_anchors()
        with torch.no_grad():
            image = torch.from_numpy(preprocess_input(image).transpose(2, 0, 1)).unsqueeze(0).type(torch.FloatTensor)
            if self.cuda:
                self.anchors = self.anchors.cuda()
                image = image.cuda()
            loc, conf, landms = self.net(image)
            boxes = decode(loc.data.squeeze(0), self.anchors, self.cfg['variance'])
            conf = conf.data.squeeze(0)[:, 1:2]
            landms = decode_landm(landms.data.squeeze(0), self.anchors, self.cfg['variance'])
            boxes_conf_landms = torch.cat([boxes, conf, landms], -1)
            boxes_conf_landms = non_max_suppression(boxes_conf_landms, self.confidence)
            if len(boxes_conf_landms) <= 0:
                return np.array([]),np.array([])
            if self.letterbox_image:
                boxes_conf_landms = facedetectrf_correct_boxes(boxes_conf_landms, \
                                                             np.array([self.input_shape[0], self.input_shape[1]]),
                                                             np.array([im_height, im_width]))
        boxes_conf_landms[:, :4] = boxes_conf_landms[:, :4] * scale
        boxes_conf_landms[:, 5:] = boxes_conf_landms[:, 5:] * scale_for_landmarks
        for i in range(len(boxes_conf_landms)):
            b_box = []
            point = []
            for b_i in range(5):
                b_box.append(boxes_conf_landms[i][b_i])
            for p_i in range(10):
                point.append(boxes_conf_landms[i][5+p_i])
            total_boxes.append(b_box)
            points.append(point)
        total_boxes = np.array(total_boxes)
        points = np.array(points)
        return total_boxes,points
