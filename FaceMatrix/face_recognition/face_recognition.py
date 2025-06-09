from __future__ import division
import os
import os.path as osp
import cv2
import onnxruntime
import onnx
from onnx import numpy_helper
import numpy as np

class FaceDetectORT:
    def __init__(self, model_path, cpu=False):
        self.model_path = model_path
        if cpu:
            self.providers = ['CPUExecutionProvider']
        else:
            self.providers = ['CUDAExecutionProvider']

    def check(self, track='g'):
        #default is cfat
        max_model_size_mb=1024
        max_feat_dim=512
        max_time_cost=15
        if track.startswith('m'):
            max_model_size_mb=1024
            max_feat_dim=512
            max_time_cost=10
        elif track.startswith('g'):
            max_model_size_mb=1024
            max_feat_dim=1024
            max_time_cost=20
        elif track.startswith('c'):
            max_model_size_mb = 1024
            max_feat_dim = 512
            max_time_cost = 15
        elif track.startswith('u'):
            max_model_size_mb=1024
            max_feat_dim=1024
            max_time_cost=30
        else:
            print("track not found")
        if not os.path.exists(self.model_path):
            print("model_path not exists")
        if not os.path.isdir(self.model_path):
            print("model_path should be directory")
        onnx_files = []
        for _file in os.listdir(self.model_path):
            if _file.endswith('.onnx'):
                onnx_files.append(osp.join(self.model_path, _file))
        if len(onnx_files)==0:
            print("do not have onnx files")
        self.model_file = sorted(onnx_files)[-1]
        try:
            session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)
        except:
            print("load onnx failed")
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        if len(input_shape)!=4:
            print("length of input_shape should be 4")
        if not isinstance(input_shape[0], str):
            #return "input_shape[0] should be str to support batch-inference"
            model = onnx.load(self.model_file)
            model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
            new_model_file = osp.join(self.model_path, 'zzzzrefined.onnx')
            onnx.save(model, new_model_file)
            self.model_file = new_model_file
            try:
                session = onnxruntime.InferenceSession(self.model_file, providers=self.providers)
            except:
                print("load onnx failed")
            input_cfg = session.get_inputs()[0]
            input_shape = input_cfg.shape
        self.image_size = tuple(input_shape[2:4][::-1])
        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        if len(output_names)!=1:
            print("number of output nodes should be 1")
        self.session = session
        self.input_name = input_name
        self.output_names = output_names
        model = onnx.load(self.model_file)
        graph = model.graph
        if len(graph.node)<8:
            print("too small onnx graph")
        input_size = (112,112)
        self.crop = None
        if track=='c':
            crop_file = osp.join(self.model_path, 'crop.txt')
            if osp.exists(crop_file):
                lines = open(crop_file,'r').readlines()
                if len(lines)!=6:
                    print("crop.txt should contain 6 lines")
                lines = [int(x) for x in lines]
                self.crop = lines[:4]
                input_size = tuple(lines[4:6])
        if input_size!=self.image_size:
            print("input-size is inconsistant with onnx model input, %s vs %s"%(input_size, self.image_size))
        self.model_size_mb = os.path.getsize(self.model_file) / float(1024*1024)
        if self.model_size_mb > max_model_size_mb:
            print("max model size exceed, given %.3f-MB"%self.model_size_mb)
        input_mean = None
        input_std = None
        if track=='c':
            pn_file = osp.join(self.model_path, 'pixel_norm.txt')
            if osp.exists(pn_file):
                lines = open(pn_file,'r').readlines()
                if len(lines)!=2:
                    print("pixel_norm.txt should contain 2 lines")
                input_mean = float(lines[0])
                input_std = float(lines[1])
        if input_mean is not None or input_std is not None:
            if input_mean is None or input_std is None:
                print("please set input_mean and input_std simultaneously")
        else:
            find_sub = False
            find_mul = False
            for nid, node in enumerate(graph.node[:8]):
                # print(nid, node.name)
                if node.name.startswith('Sub') or node.name.startswith('_minus'):
                    find_sub = True
                if node.name.startswith('Mul') or node.name.startswith('_mul') or node.name.startswith('Div'):
                    find_mul = True
            if find_sub and find_mul:
                #mxnet arcface model
                input_mean = 0.0
                input_std = 1.0
            else:
                input_mean = 127.5
                input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        for initn in graph.initializer:
            weight_array = numpy_helper.to_array(initn)
            dt = weight_array.dtype
            if dt.itemsize<4:
                print('invalid weight type - (%s:%s)' % (initn.name, dt.name))

    def forward(self, img):
        input_size = self.image_size
        if self.crop is not None:
            nimg = img[self.crop[1]:self.crop[3],self.crop[0]:self.crop[2],:]
            if nimg.shape[0]!=input_size[1] or nimg.shape[1]!=input_size[0]:
                nimg = cv2.resize(nimg, input_size)
            img = nimg
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size, (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def getFeature(self, test_img):
        if test_img is None:
            return None
        else:
            test_img = np.transpose(test_img, (1, 2, 0))
            feat = self.forward(test_img)
            return feat




