# -*- coding: utf-8 -*-
# @Organization  : rflm106det.ai
# @Author        : Jia Guo
# @Time          : 2021-05-04
# @Function      : 


from __future__ import division

import glob
import os.path as osp

import numpy as np
import onnxruntime
from numpy.linalg import norm

from ..model_zoo import model_zoo
from ..utils import DEFAULT_MP_NAME, ensure_available
from .common import Face
import face_preprocess
import cv2

__all__ = ['FaceAnalysis']

class FaceAnalysis:
    def __init__(self, name=DEFAULT_MP_NAME, model_path='face_detect_model/rf_landmark106', allowed_modules=None, **kwargs):
        onnxruntime.set_default_logger_severity(3)
        self.models = {}
        # 模型路径
        self.model_dir = model_path
        # onnx_files文件绝对路径
        onnx_files = glob.glob(osp.join(self.model_dir, '*.onnx'))
        onnx_files = sorted(onnx_files)
        for onnx_file in onnx_files:
            model = model_zoo.get_model(onnx_file, **kwargs)
            if model is None:
                print('model not recognized:', onnx_file)
            elif allowed_modules is not None and model.taskname not in allowed_modules:
                del model
            elif model.taskname not in self.models and (allowed_modules is None or model.taskname in allowed_modules):
                self.models[model.taskname] = model
            else:
                del model
        assert 'detection' in self.models
        self.det_model = self.models['detection']


    def prepare(self, ctx_id, det_thresh=0.5, det_size=(640, 640)):
        self.det_thresh = det_thresh
        assert det_size is not None
        print('set det-size:', det_size)
        self.det_size = det_size
        for taskname, model in self.models.items():
            if taskname=='detection':
                model.prepare(ctx_id, input_size=det_size, det_thresh=det_thresh)
            else:
                model.prepare(ctx_id)

    # 返回检测到的人脸，max_num为检测的人脸数目，0代表全部检测
    def get_input_rf_106lm(self, img, max_num=1):
        bboxes, kpss = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes.shape[0] == 0:
            return None, None
        ret = [] # 如果是多个人脸的话就有用了，for循环遍历
        bbox_list = []
        det_score_list = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            bbox_list.append(bbox)
            det_score = bboxes[i, 4]
            det_score_list.append(det_score)
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            # 这部分是106个关键点部分，self.models={'detectiom':<>,'landmark_2d_106':<>}
            for taskname, model in self.models.items():
                if taskname=='detection':
                    continue
                model.get(img, face)
            ret.append(face)
        # 由于只有一个人脸，所以只是取出第一个就可以
        if len(ret) !=0:
            landmark = ret[0]
            # 进行人脸对齐
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark['landmark_2d_106'][35] + landmark['landmark_2d_106'][39] + landmark['landmark_2d_106'][38]) / 3
            landmark5[1] = (landmark['landmark_2d_106'][89] + landmark['landmark_2d_106'][93] + landmark['landmark_2d_106'][88]) / 3
            landmark5[2] = landmark['landmark_2d_106'][86]
            landmark5[3] = (landmark['landmark_2d_106'][52] + landmark['landmark_2d_106'][65]) / 2
            landmark5[4] = (landmark['landmark_2d_106'][69] + landmark['landmark_2d_106'][61]) / 2
            nimg = face_preprocess.preprocess(img, bbox_list[0], landmark5, image_size='112,112')
            # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            # aligned = np.transpose(nimg, (2, 0, 1))
            return nimg, bbox_list[0]
        else:
            return None, None

    def get_input_rf_106lm_addRot(self, img, max_num=1):
        detect_count = 1
        bboxes_count_one, kpss_count_one = self.det_model.detect(img,
                                             max_num=max_num,
                                             metric='default')
        if bboxes_count_one.shape[0] == 0:
            return None, None
        if (kpss_count_one[0][0][1] > kpss_count_one[0][3][1]) and (
                kpss_count_one[0][0][1] > kpss_count_one[0][4][1]) \
                and (kpss_count_one[0][1][1] > kpss_count_one[0][3][1]) and (
                kpss_count_one[0][1][1] > kpss_count_one[0][4][1]):
            rot_face_Img = np.rot90(img, 2)
            #print("旋转180度")
            detect_count = 2
            bboxes_count_two, kpss_count_two = self.det_model.detect(rot_face_Img,
                                                                    max_num=max_num,
                                                                    metric='default')
            if bboxes_count_two.shape[0] == 0:
                return None, None
            img = rot_face_Img
        elif (kpss_count_one[0][0][0] < kpss_count_one[0][3][0]) and (kpss_count_one[0][0][0] < kpss_count_one[0][4][0]) \
            and (kpss_count_one[0][1][0] < kpss_count_one[0][3][0]) and (kpss_count_one[0][1][0] < kpss_count_one[0][4][0]) and (
            kpss_count_one[0][0][1] > kpss_count_one[0][2][1]):
            rot_face_Img = np.rot90(img, 3)
            #print("旋转270度")
            detect_count = 2
            bboxes_count_two, kpss_count_two = self.det_model.detect(rot_face_Img,
                                                                     max_num=max_num,
                                                                     metric='default')
            if bboxes_count_two.shape[0] == 0:
                return None, None
            img = rot_face_Img
        elif (kpss_count_one[0][3][0] < kpss_count_one[0][0][0]) and (kpss_count_one[0][3][0] < kpss_count_one[0][1][0]) \
            and (kpss_count_one[0][4][0] < kpss_count_one[0][0][0]) and (kpss_count_one[0][4][0] < kpss_count_one[0][1][0]) and (
            kpss_count_one[0][1][1] > kpss_count_one[0][2][1]):
            rot_face_Img = np.rot90(img, 1)
            #print("旋转180度")
            detect_count = 2
            bboxes_count_two, kpss_count_two = self.det_model.detect(rot_face_Img,
                                                                     max_num=max_num,
                                                                     metric='default')
            if bboxes_count_two.shape[0] == 0:
                return None, None
            img = rot_face_Img
        if detect_count == 1:
            bboxes = bboxes_count_one
            kpss = kpss_count_one
        elif detect_count == 2:
            bboxes = bboxes_count_two
            kpss = kpss_count_two
        # 开始真正使用lanmark106
        ret = []  # 如果是多个人脸的话就有用了，for循环遍历
        bbox_list = []
        det_score_list = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            bbox_list.append(bbox)
            det_score = bboxes[i, 4]
            det_score_list.append(det_score)
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            # 这部分是106个关键点部分，self.models={'detectiom':<>,'landmark_2d_106':<>}
            for taskname, model in self.models.items():
                if taskname == 'detection':
                    continue
                model.get(img, face)
            ret.append(face)
        # 由于只有一个人脸，所以只是取出第一个就可以
        if len(ret) != 0:
            landmark = ret[0]
            # 进行人脸对齐
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark['landmark_2d_106'][35] + landmark['landmark_2d_106'][39] +
                            landmark['landmark_2d_106'][38]) / 3
            landmark5[1] = (landmark['landmark_2d_106'][89] + landmark['landmark_2d_106'][93] +
                            landmark['landmark_2d_106'][88]) / 3
            landmark5[2] = landmark['landmark_2d_106'][86]
            landmark5[3] = (landmark['landmark_2d_106'][52] + landmark['landmark_2d_106'][65]) / 2
            landmark5[4] = (landmark['landmark_2d_106'][69] + landmark['landmark_2d_106'][61]) / 2
            nimg = face_preprocess.preprocess(img, bbox_list[0], landmark5, image_size='112,112')
            # nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
            # aligned = np.transpose(nimg, (2, 0, 1))
            return nimg, bbox_list[0]
        else:
            return None, None


    def draw_on(self, img, faces):
        import cv2
        dimg = img.copy()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(np.int)
            color = (0, 0, 255)
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(np.int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                               2)
            if face.gender is not None and face.age is not None:
                cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

            #for key, value in face.items():
            #    if key.startswith('landmark_3d'):
            #        print(key, value.shape)
            #        print(value[0:10,:])
            #        lmk = np.round(value).astype(np.int)
            #        for l in range(lmk.shape[0]):
            #            color = (255, 0, 0)
            #            cv2.circle(dimg, (lmk[l][0], lmk[l][1]), 1, color,
            #                       2)
        return dimg

