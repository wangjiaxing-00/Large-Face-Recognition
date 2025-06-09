import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import io
import configparser
import cv2
import base64
import json
import numpy as np
from PIL import Image, ImageStat
import onnxruntime
from flask import Flask, request, jsonify, Response
import torch
# 人脸检测模型包
from face_detect.face_model_rf import FaceDetectModelRF
from face_detect.face_model_sd import FaceDetectModelSD
from face_detect.sddet.sd_init import scrfd_model_init
from face_detect.rflm106det.app import FaceAnalysis
# 人脸图像质量评估
import EQFace.model_resnet as EQFaceModel
import EQFace.utils as EQFaceUtils
from SDD_FIQA.model_network import network
# 口罩检测
from FaceMask.FaceMaskNet import yolov5_lite

# 配置文件
config = configparser.ConfigParser()
config.read("./config/algorithmPara.ini", "utf-8")
faceEngineVersion = config.get('algorithm', 'faceEngineVersion')
faceEngineName = config.get('algorithm', 'faceEngineName')
face_detect_model_flag = config.get('algorithm', 'face_detect_model_flag')
getFeatureErrRotFlag = config.getboolean('algorithm', 'getFeatureErrRotFlag')
angleRotFlag = config.getboolean('algorithm', 'angleRotFlag')
brightness_switch = config.getboolean('algorithm', 'brightness_switch')
definition_switch = config.getboolean('algorithm', 'definition_switch')
face_head_angle_switch = config.getboolean('algorithm', 'face_head_angle_switch')
face_quality_switch = config.getboolean('algorithm', 'face_quality_switch')
face_mask_switch = config.getboolean('algorithm', 'face_mask_switch')
face_quality_algorithm = config.get('algorithm', 'face_quality_algorithm')


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 检测模型
if face_detect_model_flag == "0":
    face_detect_model = FaceDetectModelRF()
elif face_detect_model_flag == "1":
    face_detect_model = FaceDetectModelSD()
    bbx_detector = scrfd_model_init('face_detect_model/sd/sd_det.onnx')
elif face_detect_model_flag == "2":
    face_detect_model = FaceAnalysis(model_path='face_detect_model/rf_landmark106',
                                     allowed_modules=['detection', 'landmark_2d_106'])
    face_detect_model.prepare(ctx_id=0, det_size=(256, 256))

# 人脸质量评估模型
if face_quality_switch == True:
    # 人脸质量评估模型-EQFace
    if face_quality_algorithm == "EQFace":
        EQFace_ad_fq = 0.05
        # 加载人脸质量评估模型参数
        EQFace_backbone_model = "./EQFace/backbone.pth"
        EQFace_quality_model = "./EQFace/quality.pth"
        BACKBONE = EQFaceModel.ResNet(num_layers=100, feature_dim=512)
        QUALITY = EQFaceModel.FaceQuality(512 * 7 * 7)
        checkpoint = torch.load(EQFace_backbone_model, map_location=DEVICE)
        EQFaceUtils.load_state_dict(BACKBONE, checkpoint)
        checkpoint = torch.load(EQFace_quality_model, map_location=DEVICE)
        EQFaceUtils.load_state_dict(QUALITY, checkpoint)
        BACKBONE.to(DEVICE)
        QUALITY.to(DEVICE)
        BACKBONE.eval()
        QUALITY.eval()
    # 人脸质量评估模型-SDD_FIQA
    elif face_quality_algorithm == "SDD_FIQA":
        SDD_FIQA_model = './SDD_FIQA/SDD_FIQA_checkpoints_r50.pth'
        SDD_FIQA_net = network(SDD_FIQA_model, DEVICE)
# 人脸角度模型
if face_head_angle_switch == True:
    fsa_net_sess1 = onnxruntime.InferenceSession('FSANet/model/fsanet-1x1-iter-688590.onnx', providers=['CUDAExecutionProvider'])
    fsa_net_sess2 = onnxruntime.InferenceSession('FSANet/model/fsanet-var-iter-688590.onnx', providers=['CUDAExecutionProvider'])
# 口罩检测
if face_mask_switch == True:
    mask_net = yolov5_lite('FaceMask/last.onnx', 'FaceMask/coco.names')


# 科学计数法
def as_num(x):
    y='{:.5f}'.format(x) # 5f表示保留5位小数点的float型
    return(y)

# 图片模糊分数映射0-1之间，参数400可以调整
def blur_map_score(x):
    return 1 - np.exp(-x / 400)
# 拉普拉斯图片模糊检测
def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# base64转换为图像,返回的是BGR格式，已经验证过，和cv2.imread()返回的类型一样，可以互用
def base64_to_image(base64_code):
    img_data = base64.b64decode(base64_code)
    img_format = img_data[0:6]
    if img_format == b'GIF89a':
        #GIF格式
        img_gif = Image.open(io.BytesIO(img_data))
        img_rgb = img_gif.convert('RGB')
        pil_to_numpy = np.array(img_rgb)
        img = cv2.cvtColor(pil_to_numpy,cv2.IMREAD_COLOR)
        return img
    else:
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

def imgRot(img,k):
    a = np.rot90(img,k)
    return a

def faceDetectRot(photoData_img,face_detect_model_flag,getFeatureErrRotFlag,angleRotFlag,face_detect_model):
    if face_detect_model_flag == "0":
        if angleRotFlag == True:
            img,box  = face_detect_model.get_input_facedetectrf_addRot(photoData_img)
        else:
            img,box  = face_detect_model.get_input_facedetectrf(photoData_img)
    elif face_detect_model_flag == "1":
        if angleRotFlag == True:
            img,box = face_detect_model.get_input_facedetectsd_addRot(photoData_img, bbx_detector)
        else:
            img,box = face_detect_model.get_input_facedetectsd(photoData_img, bbx_detector)
    elif face_detect_model_flag == "2":
        if angleRotFlag == True:
            img,box = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
        else:
            img,box = face_detect_model.get_input_rf_106lm(photoData_img)

    if img is None:
        if getFeatureErrRotFlag == True:
            rotImg = imgRot(photoData_img, 2)
            if face_detect_model_flag == "0":
                if angleRotFlag == True:
                    img,box = face_detect_model.get_input_facedetectrf_addRot(rotImg)
                else:
                    img,box = face_detect_model.get_input_facedetectrf(rotImg)
            elif face_detect_model_flag == "1":
                if angleRotFlag == True:
                    img,box = face_detect_model.get_input_facedetectsd_addRot(rotImg, bbx_detector)
                else:
                    img,box = face_detect_model.get_input_facedetectsd(rotImg, bbx_detector)
            elif face_detect_model_flag == "2":
                if angleRotFlag == True:
                    img,box = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
                else:
                    img,box = face_detect_model.get_input_rf_106lm(photoData_img)
            if img is None:
                rotImg = imgRot(rotImg, 1)
                if face_detect_model_flag == "0":
                    if angleRotFlag == True:
                        img,box = face_detect_model.get_input_facedetectrf_addRot(rotImg)
                    else:
                        img,box = face_detect_model.get_input_facedetectrf(rotImg)
                elif face_detect_model_flag == "1":
                    if angleRotFlag == True:
                        img,box = face_detect_model.get_input_facedetectsd_addRot(rotImg, bbx_detector)
                    else:
                        img,box = face_detect_model.get_input_facedetectsd(rotImg, bbx_detector)
                elif face_detect_model_flag == "2":
                    if angleRotFlag == True:
                        img,box = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
                    else:
                        img,box = face_detect_model.get_input_rf_106lm(photoData_img)
                if img is None:
                    rotImg = imgRot(rotImg, 2)
                    if face_detect_model_flag == "0":
                        if angleRotFlag == True:
                            img,box = face_detect_model.get_input_facedetectrf_addRot(rotImg)
                        else:
                            img,box = face_detect_model.get_input_facedetectrf(rotImg)
                    elif face_detect_model_flag == "1":
                        if angleRotFlag == True:
                            img,box = face_detect_model.get_input_facedetectsd_addRot(rotImg, bbx_detector)
                        else:
                            img,box = face_detect_model.get_input_facedetectsd(rotImg, bbx_detector)
                    elif face_detect_model_flag == "2":
                        if angleRotFlag == True:
                            img,box = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
                        else:
                            img,box = face_detect_model.get_input_rf_106lm(photoData_img)
                    if img is None:
                        return None,None
                    else:
                        return img,box
                else:
                    return img,box
            else:
                return img,box
        else:
            return None,None
    else:
        return img,box


# 图像质量返回
def return_face_quality_result(code, message, face_quality, face_brightness, face_definition, face_angle, face_mask, face_base64):
    result = {"code": code,
              "message": message,
              "face_quality": face_quality,
              "face_brightness": face_brightness,
              "face_definition": face_definition,
              "face_angle": face_angle,
              "face_mask": face_mask,
              "face_base64": face_base64
              }
    return Response(json.dumps(result), mimetype='application/json')


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def hello_world():
    return 'Hello World!'

# 人脸图像质量
@app.route('/faceQuality', methods=['POST'])
def faceQuality():
    try:
        # 接收图像的base64
        query_image_base64 = request.json['query_image_base64']
        if len(query_image_base64) == 0:
            code = 10001
            message = "入参base64为空"
            quality = None
            brightness = None
            definition = None
            angle = None
            mask = None
            face_base64 = None
            return return_face_quality_result(code, message, quality, brightness, definition, angle, mask, face_base64)
        try:
            input_img = base64_to_image(query_image_base64)
        except Exception as e:
            code = 10001
            message = "入参base64异常"
            quality = None
            brightness = None
            definition = None
            angle = None
            mask = None
            face_base64 = None
            return return_face_quality_result(code, message, quality, brightness, definition, angle, mask, face_base64)
        try:
            # 人脸检测
            aligned_face, bbox = faceDetectRot(input_img, face_detect_model_flag, getFeatureErrRotFlag, angleRotFlag,
                                              face_detect_model)
            if aligned_face is None:
                code = 10060
                message = "人脸检测失败"
                quality = None
                brightness = None
                definition = None
                angle = None
                mask = None
                face_base64 = None
                return return_face_quality_result(code, message, quality, brightness, definition, angle, mask, face_base64)
            else:
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[2]
                y2 = bbox[3]
                w = x2 - x1
                h = y2 - y1
                img_h, img_w, _ = np.shape(input_img)
                # 光照开关开启
                if brightness_switch == True:
                    # 光照识别(含图像增强后)
                    # 将图像转换为灰度图像，这个是BGR格式数据
                    gray_image = cv2.cvtColor(input_img[int(y1):int(y2), int(x1):int(x2), :], cv2.COLOR_BGR2GRAY)
                    brightness = as_num(gray_image.mean() / 255)
                else:
                    brightness = None
                # 人脸图像清晰度开关开启
                if definition_switch == True:
                    # 人脸图像清晰度检测，这个是BGR格式数据
                    definition = as_num(blur_map_score(variance_of_laplacian(input_img[int(y1):int(y2), int(x1):int(x2), :])))
                else:
                    definition = None
                # 人脸角度评估开关开启
                if face_head_angle_switch == True:
                    # pitch:点头  yaw:摇头  roll:歪头, 这个是BGR格式数据
                    #  pitch:点头(上正，下负，建议值±30以内)  yaw:摇头(左正，右负，建议值±45以内)   roll:歪头（左负，右正，建议值±30以内）
                    ad_angle = 0.6
                    xw1_angle = max(int(x1 - ad_angle * w), 0)
                    yw1_angle = max(int(y1 - ad_angle * h), 0)
                    xw2_angle = min(int(x2 + ad_angle * w), img_w - 1)
                    yw2_angle = min(int(y2 + ad_angle * h), img_h - 1)
                    face_angle_roi = input_img[yw1_angle:yw2_angle + 1, xw1_angle:xw2_angle + 1, :]
                    face_angle_roi = cv2.resize(face_angle_roi, (64, 64))
                    face_angle_roi = face_angle_roi.transpose((2, 0, 1))
                    face_angle_roi = np.expand_dims(face_angle_roi, axis=0)
                    face_angle_roi = (face_angle_roi - 127.5) / 128
                    face_angle_roi = face_angle_roi.astype(np.float32)
                    fsa_net_res1 = fsa_net_sess1.run(["output"], {"input": face_angle_roi})[0]
                    fsa_net_res2 = fsa_net_sess2.run(["output"], {"input": face_angle_roi})[0]
                    yaw, pitch, roll = np.mean(np.vstack((fsa_net_res1, fsa_net_res2)), axis=0)
                    angle = {
                        'yaw': as_num(yaw),
                        'pitch': as_num(pitch),
                        'roll': as_num(roll)
                    }
                else:
                    angle = None
                # 人脸质量评估开关开启
                if face_quality_switch == True:
                    # 人脸质量评估，这个是BGR格式数据
                    if face_quality_algorithm == "EQFace":
                        xw1_fq = max(int(x1 - EQFace_ad_fq * w), 0)
                        yw1_fq = max(int(y1 - EQFace_ad_fq * h), 0)
                        xw2_fq = min(int(x2 + EQFace_ad_fq * w), img_w - 1)
                        yw2_fq = min(int(y2 + EQFace_ad_fq * h), img_h - 1)
                        cropImg_fq = input_img[yw1_fq:yw2_fq + 1, xw1_fq:xw2_fq + 1, :]
                        quality = as_num(EQFaceUtils.get_face_quality(BACKBONE, QUALITY, DEVICE, cropImg_fq)[0])
                    elif face_quality_algorithm == "SDD_FIQA":
                        # # 第1种写法：
                        # cropImg_fq = input_img[int(y1):int(y2), int(x1):int(x2), :]
                        # # resize为(112,112)
                        # cropImg_fq = cv2.resize(cropImg_fq, (112, 112))
                        # # BGR格式转换为RGB
                        # # 等价于cv2.cvtColor(cropImg_fq, cv2.COLOR_BGR2RGB)
                        # cropImg_fq = cropImg_fq[..., ::-1]
                        # # 由(112,112,3)转换为(3,112,112)
                        # cropImg_fq = cropImg_fq.transpose((2, 0, 1))
                        # # 图像数据归一化，将像素缩放到[-1,1]之间
                        # cropImg_fq = (cropImg_fq - 127.5) / 128.0
                        # cropImg_fq = cropImg_fq.astype(np.float32)
                        # cropImg_fq = cropImg_fq[np.newaxis, ]
                        # cropImg_fq_tensor = torch.from_numpy(cropImg_fq).to(DEVICE)
                        # face_quality = (SDD_FIQA_net(cropImg_fq_tensor).cpu().tolist()[0][0])/100
                        # 第2种写法
                        cropImg_fq = input_img[int(y1):int(y2), int(x1):int(x2), :]
                        # resize为(112,112)
                        cropImg_fq = cv2.resize(cropImg_fq, (112, 112))
                        # BGR格式转换为RGB
                        cropImg_fq = cropImg_fq[..., ::-1]
                        # 图像数据归一化，将像素缩放到[-1,1]之间
                        cropImg_fq = (cropImg_fq - 127.5) / 128.0
                        # 先转换为tensor类型，写到GPU，然后变换顺序由(112,112,3)转换为(3,112,112)
                        # .to(torch.float32)非常重要，这样写cpu不会消耗
                        # permute(2,0,1)等价于transpose((2, 0, 1))
                        # .unsqueeze(0)扩充一列,由(3,112,112)变成(1,3,112,112)
                        cropImg_fq_tensor = torch.from_numpy(cropImg_fq).cuda().permute(2, 0, 1).to(torch.float32).unsqueeze(0)
                        quality = as_num((SDD_FIQA_net(cropImg_fq_tensor).cpu().tolist()[0][0])/100)
                else:
                    quality = None
                # 口罩开关开启
                if face_mask_switch == True:
                    cropImg_mask = input_img[int(y1):int(y2), int(x1):int(x2), :]
                    # 1代表有口罩，0代表没有
                    mask = str(mask_net.detect(cropImg_mask))
                else:
                    mask = None
                # 将裁剪对其后的图片转换为base64
                crop_align_cvt_image = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                crop_align_imgPIL = Image.fromarray(crop_align_cvt_image)
                crop_align_buffered = io.BytesIO()
                crop_align_imgPIL.save(crop_align_buffered, format="PNG")
                face_base64 = base64.b64encode(crop_align_buffered.getvalue()).decode('utf-8')
                code = 0
                message = "成功"
                return return_face_quality_result(code, message, quality, brightness, definition, angle, mask, face_base64)
        except Exception as e:
            code = 20000
            message = str(e)
            quality = None
            brightness = None
            definition = None
            angle = None
            mask = None
            face_base64 = None
            return return_face_quality_result(code, message, quality, brightness, definition, angle, mask, face_base64)
    except Exception as e:
        code = 20000
        message = str(e)
        quality = None
        brightness = None
        definition = None
        angle = None
        mask = None
        face_base64 = None
        return return_face_quality_result(code, message, quality, brightness, definition, angle, mask, face_base64)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=False, debug=False)
    #app.run()











