from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import mxnet as mx
import cv2
from sklearn import preprocessing
from face_detect.rfdet.facedetectrf import FaceDetectRF
import face_preprocess
import time

def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

class FaceDetectModelRF:
  def __init__(self):
    detector = FaceDetectRF()
    self.detector = detector

  def get_input_facedetectrf(self, face_img):
    ret = self.detector.detect_image(face_img)
    if ret is None:
        return None
    bbox_before, points_before = ret
    if bbox_before.shape[0]==0:
        return None
    bbox = bbox_before[0,0:4]
    points=[]
    points.append(points_before[0][0])
    points.append(points_before[0][2])
    points.append(points_before[0][4])
    points.append(points_before[0][6])
    points.append(points_before[0][8])
    points.append(points_before[0][1])
    points.append(points_before[0][3])
    points.append(points_before[0][5])
    points.append(points_before[0][7])
    points.append(points_before[0][9])
    points = np.array(points)
    points = points.reshape((2,5)).T
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    return aligned

  def get_input_facedetectrf_addRot(self,face_img):
    detect_count = 1
    ret = self.detector.detect_image(face_img)
    if ret is None:
      return None
    bbox_before_count_one, points_before_count_one = ret
    if bbox_before_count_one.shape[0] == 0:
      return None
    bbox_count_one = bbox_before_count_one[0, 0:4]
    if (points_before_count_one[0][1] > points_before_count_one[0][7]) and (points_before_count_one[0][1] > points_before_count_one[0][9]) \
            and (points_before_count_one[0][3] > points_before_count_one[0][7]) and (points_before_count_one[0][3] > points_before_count_one[0][9]):
      rot_face_Img = np.rot90(face_img, 2)
      detect_count = 2
      ret = self.detector.detect_image(rot_face_Img)
      if ret is None:
        return None
      bbox_before_count_two, points_before_count_two = ret
      if bbox_before_count_two.shape[0] == 0:
        return None
      bbox_count_two = bbox_before_count_two[0, 0:4]
      face_img = rot_face_Img
    elif (points_before_count_one[0][0] < points_before_count_one[0][6]) and (points_before_count_one[0][0] < points_before_count_one[0][8]) \
            and (points_before_count_one[0][2] < points_before_count_one[0][6]) and (points_before_count_one[0][2] < points_before_count_one[0][8]) and (
            points_before_count_one[0][1] > points_before_count_one[0][5]):
      rot_face_Img = np.rot90(face_img, 3)
      detect_count = 2
      ret = self.detector.detect_image(rot_face_Img)
      if ret is None:
        return None
      bbox_before_count_two, points_before_count_two = ret
      if bbox_before_count_two.shape[0] == 0:
        return None
      bbox_count_two = bbox_before_count_two[0, 0:4]
      face_img = rot_face_Img
    elif (points_before_count_one[0][6] < points_before_count_one[0][0]) and (points_before_count_one[0][6] < points_before_count_one[0][2]) \
            and (points_before_count_one[0][8] < points_before_count_one[0][0]) and (points_before_count_one[0][8] < points_before_count_one[0][2]) and (
            points_before_count_one[0][3] > points_before_count_one[0][5]):
      rot_face_Img = np.rot90(face_img, 1)
      detect_count = 2
      ret = self.detector.detect_image(rot_face_Img)
      if ret is None:
        return None
      bbox_before_count_two, points_before_count_two = ret
      if bbox_before_count_two.shape[0] == 0:
        return None
      bbox_count_two = bbox_before_count_two[0, 0:4]
      face_img = rot_face_Img
    if detect_count == 1:
      bbox = bbox_count_one
      points_before = points_before_count_one
    elif detect_count == 2:
      bbox = bbox_count_two
      points_before = points_before_count_two
    points = []
    points.append(points_before[0][0])
    points.append(points_before[0][2])
    points.append(points_before[0][4])
    points.append(points_before[0][6])
    points.append(points_before[0][8])
    points.append(points_before[0][1])
    points.append(points_before[0][3])
    points.append(points_before[0][5])
    points.append(points_before[0][7])
    points.append(points_before[0][9])
    points = np.array(points)
    points = points.reshape((2, 5)).T
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    return aligned

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = preprocessing.normalize(embedding).flatten()
    return embedding

  def get_ga(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.ga_model.forward(db, is_train=False)
    ret = self.ga_model.get_outputs()[0].asnumpy()
    g = ret[:,0:2].flatten()
    gender = np.argmax(g)
    a = ret[:,2:202].reshape( (100,2) )
    a = np.argmax(a, axis=1)
    age = int(sum(a))
    return gender, age

