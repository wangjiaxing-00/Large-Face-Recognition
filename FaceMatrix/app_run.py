import os
import io
import datetime
import configparser
import json
import cv2
import logging
import logging.handlers
import numpy as np
import base64
from flask import Flask, request, jsonify, Response
from sklearn import preprocessing
from PIL import Image, ImageStat
from face_detect.sddet.sd_init import scrfd_model_init
from face_detect.face_model_rf import FaceDetectModelRF
from face_detect.face_model_sd import FaceDetectModelSD
from face_recognition.face_recognition import FaceDetectORT
from license.verLicense import is_license_check, is_license_valid
from face_detect.rflm106det.app import FaceAnalysis
from face_quality_api import getFaceQuality
from concurrent.futures import ThreadPoolExecutor


config = configparser.ConfigParser()
config.read("./config/algorithmPara.ini", "utf-8")

faceEngineVersion = config.get('algorithm', 'faceEngineVersion')
faceEngineName = config.get('algorithm', 'faceEngineName')
face_detect_model_flag = config.get('algorithm', 'face_detect_model_flag')
getFeatureErrRotFlag = config.getboolean('algorithm', 'getFeatureErrRotFlag')
angleRotFlag = config.getboolean('algorithm', 'angleRotFlag')
imgAug = config.getboolean('algorithm', 'imgAug')
brightnessLowThreshold = config.getfloat('algorithm', 'brightnessLowThreshold')
brightnessHighThreshold = config.getfloat('algorithm', 'brightnessHighThreshold')
error_log_flag = config.getboolean('log', 'error_log_flag')
errorLogPath = config.get('log', 'errorLogPath')
info_log_flag = config.getboolean('log', 'info_log_flag')
infoLogPath = config.get('log', 'infoLogPath')
licnese_enc = config.get('license', 'enc')
licnese_aes = config.get('license', 'aes')
licnese_private = config.get('license', 'private')
is_invalid = config.getboolean('license', 'is_invalid')

logger_error = logging.getLogger('error_logger')
logger_info = logging.getLogger('info_logger')
logger_error.setLevel(level=logging.ERROR)
logger_info.setLevel(level=logging.INFO)
# log_error_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_error.log"
# log_info_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_info.log"
# handler_error = logging.handlers.RotatingFileHandler(os.path.join(errorLogPath, log_error_filename), maxBytes=200*1024*1024, backupCount=100)
# handler_info = logging.handlers.RotatingFileHandler(os.path.join(infoLogPath, log_info_filename), maxBytes=200*1024*1024, backupCount=100)
# formatter_error = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# formatter_info = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler_error.setFormatter(formatter_error)
# handler_info.setFormatter(formatter_info)
# logger_error.addHandler(handler_error)
# logger_info.addHandler(handler_info)


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


# 识别模型
face_recognition = FaceDetectORT("face_recognition_model/500w_r100_128D")
face_recognition.check("g")

license_check = is_license_check(licnese_enc, licnese_aes, licnese_private)
# alpha=-4.83702,beta=5.56252
def distance(embeddings1,embeddings2,distance_metric=0,alpha=-9.81675,beta=10.93322):
    if distance_metric == 0:
        diff = np.subtract(embeddings1,embeddings2)
        dist = np.sum(np.square(diff),0)
        dist = 1-(1./(1+np.exp(alpha*dist+beta)))
    elif distance_metric == 1:
        dot = np.sum(np.multiply(embeddings1,embeddings2),axis=0)
        norm = np.linalg.norm(embeddings1,axis=0)* np.linalg.norm(embeddings2,axis=0)
        similarity = dot/norm
        dist = similarity
    return dist

def as_num(x):
    y='{:.2f}'.format(x)
    return(y)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

# 转换图像到灰度，返回平均像素亮度
def brightness(image):
   im = Image.fromarray(np.uint8(image)).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def regu(num):
    if(num>255):
        return 255
    elif num < 0:
        return 0
    else :
        return num

# 修改图像的亮度，brightness取值0～2 <1表示变暗 >1表示变亮
def change_brightness(img, brightness):
    [averB, averG, averR] = np.array(cv2.mean(img))[:-1] / 3
    k = np.ones((img.shape))
    k[:, :, 0] *= averB
    k[:, :, 1] *= averG
    k[:, :, 2] *= averR
    img = img + (brightness - 1) * k
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.uint8)

# 修改图像的对比度,coefficent>0, <1降低对比度,>1提升对比度 建议0-2
def change_contrast(img, coefficent):
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.mean(img)[0]
    graynew = m + coefficent * (imggray - m)
    img1 = np.zeros(img.shape, np.float32)
    k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
    img1[:, :, 0] = img[:, :, 0] * k
    img1[:, :, 1] = img[:, :, 1] * k
    img1[:, :, 2] = img[:, :, 2] * k
    img1[img1 > 255] = 255
    img1[img1 < 0] = 0
    return img1.astype(np.uint8)


def single_img_clahe(test):
    try:
        B, G, R = cv2.split(test)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        clahe_B = clahe.apply(B)
        clahe_G = clahe.apply(G)
        clahe_R = clahe.apply(R)
        clahe_test = cv2.merge((clahe_B, clahe_G, clahe_R))
        return clahe_test
    except:
        return test


# 图像增强
def imageAugmentation(image):
    if imgAug == True:
        brightnessScore = brightness(image)
        if (brightnessScore <= brightnessLowThreshold) :
            image = single_img_clahe(image)
            return image
        elif (brightnessScore >= brightnessHighThreshold):
            image_temp = change_contrast(image, 1.14)
            image = change_brightness(image_temp, 0.05)
            return image
        else:
            return image
    else:
        return image


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
        #img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # COLOR_RGB2BGR
        return img

def imgRot(img,k):
    a = np.rot90(img,k)
    return a

def faceDetectFixedCrop(img):
    size = img.shape
    w = size[1]
    h = size[0]
    delta_w = (8/10) * w
    delta_h = (8/10) * h
    start_x = int((1/10) * w)
    start_y = int((1/10) * h)
    stop_x = int(start_x + delta_w)
    stop_y = int(start_y + delta_h)
    cropImg = img[start_y:stop_y,start_x:stop_x]
    return cropImg

def faceDetectRot(photoData_img,face_detect_model_flag,getFeatureErrRotFlag,angleRotFlag,face_detect_model):
    if face_detect_model_flag == "0":
        if angleRotFlag == True:
            img = face_detect_model.get_input_facedetectrf_addRot(photoData_img)
        else:
            img = face_detect_model.get_input_facedetectrf(photoData_img)
    elif face_detect_model_flag == "1":
        if angleRotFlag == True:
            img = face_detect_model.get_input_facedetectsd_addRot(photoData_img, bbx_detector)
        else:
            img = face_detect_model.get_input_facedetectsd(photoData_img, bbx_detector)
    elif face_detect_model_flag == "2":
        if angleRotFlag == True:
            img = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
        else:
            img = face_detect_model.get_input_rf_106lm(photoData_img)

    if img is None:
        if getFeatureErrRotFlag == True:
            rotImg = imgRot(photoData_img, 2)
            if face_detect_model_flag == "0":
                if angleRotFlag == True:
                    img = face_detect_model.get_input_facedetectrf_addRot(rotImg)
                else:
                    img = face_detect_model.get_input_facedetectrf(rotImg)
            elif face_detect_model_flag == "1":
                if angleRotFlag == True:
                    img = face_detect_model.get_input_facedetectsd_addRot(rotImg, bbx_detector)
                else:
                    img = face_detect_model.get_input_facedetectsd(rotImg, bbx_detector)
            elif face_detect_model_flag == "2":
                if angleRotFlag == True:
                    img = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
                else:
                    img = face_detect_model.get_input_rf_106lm(photoData_img)
            if img is None:
                rotImg = imgRot(rotImg, 1)
                if face_detect_model_flag == "0":
                    if angleRotFlag == True:
                        img = face_detect_model.get_input_facedetectrf_addRot(rotImg)
                    else:
                        img = face_detect_model.get_input_facedetectrf(rotImg)
                elif face_detect_model_flag == "1":
                    if angleRotFlag == True:
                        img = face_detect_model.get_input_facedetectsd_addRot(rotImg, bbx_detector)
                    else:
                        img = face_detect_model.get_input_facedetectsd(rotImg, bbx_detector)
                elif face_detect_model_flag == "2":
                    if angleRotFlag == True:
                        img = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
                    else:
                        img = face_detect_model.get_input_rf_106lm(photoData_img)
                if img is None:
                    rotImg = imgRot(rotImg, 2)
                    if face_detect_model_flag == "0":
                        if angleRotFlag == True:
                            img = face_detect_model.get_input_facedetectrf_addRot(rotImg)
                        else:
                            img = face_detect_model.get_input_facedetectrf(rotImg)
                    elif face_detect_model_flag == "1":
                        if angleRotFlag == True:
                            img = face_detect_model.get_input_facedetectsd_addRot(rotImg, bbx_detector)
                        else:
                            img = face_detect_model.get_input_facedetectsd(rotImg, bbx_detector)
                    elif face_detect_model_flag == "2":
                        if angleRotFlag == True:
                            img = face_detect_model.get_input_rf_106lm_addRot(photoData_img)
                        else:
                            img = face_detect_model.get_input_rf_106lm(photoData_img)
                    if img is None:
                        # photoData_img---<class 'numpy.ndarry'>转化为base64
                        data = cv2.imencode('.jpg', photoData_img)[1]
                        image_bytes = data.tobytes()
                        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                        # logger_fb_status.error("face not detect - crop image - base64 - " + image_base64)
                        cropImg = faceDetectFixedCrop(photoData_img)
                        nimg = face_preprocess.preprocess(cropImg,image_size='112,112')
                        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                        img = np.transpose(nimg, (2, 0, 1))
                        return img
                        # return None
                    else:
                        return img
                else:
                    return img
            else:
                return img
        else:
            return None
    else:
        return img


# 比对返回
def return_compare_result(code, message, ainumber, similarity):
    result = {"code": code,
              "message": message,
              "ainumber": ainumber,
              "similarity": similarity
              }
    return Response(json.dumps(result), mimetype='application/json')

# 提取特征值返回
def return_feature_result(code, message, ainumber, data):
    result = {"code": code,
              "message": message,
              "ainumber": ainumber,
              "data": data
              }
    return Response(json.dumps(result), mimetype='application/json')

# 批量特征值返回
def return_batch_feature_result(code, message, ainumber, data):
    result = {
        "code": code,
        "message": message,
        "ainumber": ainumber,
        "data": data
    }
    return Response(json.dumps(result), mimetype='application/json')

# 批量建模线程池异步处理
def handle_image(item):
    serial_number = item["serialNumber"]
    image_base64 = item["imageBase64"]
    try:
        r = feature_extract(image_base64)
        r_json = r.get_json()  # 将Response对象转换为字典
        if r_json["code"] == 0:
            return {
                "serialNumber": serial_number,
                "feature": r_json["data"]["feature"],
                "quality": r_json["data"]["quality"]
            }
        else:
            return {
                "serialNumber": serial_number,
                "reason": r_json["code"]
            }
    except Exception as e:
        return {
            "serialNumber": serial_number,
            "reason": "20000"
        }


# 图片比图片
def image_and_image_verify(query_image_base64, id_image_base64):
    try:
        # is_invalid = is_license_valid(license_check)
        # license通过
        if is_invalid:
            try:
                if info_log_flag == True:
                    logger_info.info("compareImageAndImage")
                # 首先经过图像质量评估模块
                query_code, query_message, query_face_base64 = getFaceQuality(query_image_base64)
                id_code, id_message, id_face_base64 = getFaceQuality(id_image_base64)
                # 代表code是1代表不经过图像质量评估
                if query_code == 1 and id_code == 1:
                    if (len(query_image_base64) == 0 or len(id_image_base64) == 0):
                        code = 10001
                        message = "入参base64为空"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                    try:
                        query_image = base64_to_image(query_image_base64)
                    except Exception as e:
                        code = 10001
                        message = "query入参base64异常"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                    try:
                        id_image = base64_to_image(id_image_base64)
                    except Exception as e:
                        code = 10001
                        message = "id入参base64异常"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                    # query图像增强
                    query_image_augment = imageAugmentation(query_image)
                    # query图像人脸检测
                    query_image_face_detect = faceDetectRot(query_image_augment, face_detect_model_flag,
                                                            getFeatureErrRotFlag, angleRotFlag,
                                                            face_detect_model)
                    # id图像增强
                    id_image_augment = imageAugmentation(id_image)
                    # id图像人脸检测
                    id_image_face_detect = faceDetectRot(id_image_augment, face_detect_model_flag,
                                                         getFeatureErrRotFlag, angleRotFlag,
                                                         face_detect_model)
                    if query_image_face_detect is None:
                        code = 10060
                        message = "query人脸检测失败"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                    if id_image_face_detect is None:
                        code = 10060
                        message = "id人脸检测失败"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                # 代表code是0代表经过图像质量后都合规
                elif query_code == 0 and id_code == 0:
                    query_image_face_detect = np.transpose(base64_to_image(query_face_base64), (2, 0, 1))
                    id_image_face_detect = np.transpose(base64_to_image(id_face_base64), (2, 0, 1))
                else:
                    if query_code != 0:
                        code = query_code
                        message = query_message
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                    else:
                        code = id_code
                        message = id_message
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                # 提取特征
                try:
                    f1 = face_recognition.getFeature(query_image_face_detect)
                    f2 = face_recognition.getFeature(id_image_face_detect)
                    f1 = preprocessing.normalize(f1)
                    f2 = preprocessing.normalize(f2)
                except Exception as e:
                    code = 10070
                    message = "抽取特征值失败"
                    ainumber = "6"
                    similarity = None
                    return return_compare_result(code, message, ainumber, similarity)
                # 特征比对
                try:
                    sim = distance(f1[0], f2[0], 0)
                    code = 0
                    message = "成功"
                    ainumber = "6"
                    similarity = float(as_num(sim * 100))
                    return return_compare_result(code, message, ainumber, similarity)
                except Exception as e:
                    code = 20000
                    message = str(e)
                    ainumber = "6"
                    similarity = None
                    return return_compare_result(code, message, ainumber, similarity)
            except Exception as e:
                code = 20000
                message = str(e)
                ainumber = "6"
                similarity = None
                return return_compare_result(code, message, ainumber, similarity)
        else:
            code = 20000
            message = "LICENSE过期"
            ainumber = "6"
            similarity = None
            return return_compare_result(code, message, ainumber, similarity)
    except Exception as e:
        code = 20000
        message = str(e)
        ainumber = "6"
        similarity = None
        if error_log_flag == True:
            logger_error.error("compareImageAndImage - " + str(e))
        return return_compare_result(code, message, ainumber, similarity)


# 图像与特征比对
def image_and_feature_verify(query_image_base64, id_image_feature):
    try:
        # is_invalid = is_license_valid(license_check)
        # license通过
        if is_invalid:
            try:
                if info_log_flag == True:
                    logger_info.info("compareImageAndFeature")
                if (len(id_image_feature) == 0):
                    code = 10002
                    message = "入参特征值为空"
                    ainumber = "6"
                    similarity = None
                    return return_compare_result(code, message, ainumber, similarity)
                # 首先经过图像质量评估模块
                query_code, query_message, query_face_base64 = getFaceQuality(query_image_base64)
                # 代表code是1代表不经过图像质量评估
                if query_code == 1:
                    if (len(query_image_base64) == 0):
                        code = 10001
                        message = "入参base64为空"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                    try:
                        query_image = base64_to_image(query_image_base64)
                    except Exception as e:
                        code = 10001
                        message = "入参base64异常"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                    # query图像增强
                    query_image_augment = imageAugmentation(query_image)
                    # query人脸检测
                    query_image_face_detect = faceDetectRot(query_image_augment, face_detect_model_flag,
                                                            getFeatureErrRotFlag, angleRotFlag,
                                                            face_detect_model)
                    if query_image_face_detect is None:
                        code = 10060
                        message = "人脸检测失败"
                        ainumber = "6"
                        similarity = None
                        return return_compare_result(code, message, ainumber, similarity)
                # 代表code是0代表经过图像质量后都合规
                elif query_code == 0:
                    query_image_face_detect = np.transpose(base64_to_image(query_face_base64), (2, 0, 1))
                else:
                    code = query_code
                    message = query_message
                    ainumber = "6"
                    similarity = None
                    return return_compare_result(code, message, ainumber, similarity)
                try:
                    id_image_feature_base64_bytes = id_image_feature.encode()
                    id_image_feature_bytes = base64.decodebytes(id_image_feature_base64_bytes)
                    id_image_feature_numpy = np.frombuffer(id_image_feature_bytes, dtype=np.float32)
                except Exception as e:
                    code = 10002
                    message = "入参特征值异常"
                    ainumber = "6"
                    similarity = None
                    return return_compare_result(code, message, ainumber, similarity)
                # 提取特征值
                try:
                    query_image_feature = face_recognition.getFeature(query_image_face_detect)
                    query_image_feature = preprocessing.normalize(query_image_feature)
                except Exception as e:
                    code = 10070
                    message = "抽取特征值失败"
                    ainumber = "6"
                    similarity = None
                    return return_compare_result(code, message, ainumber, similarity)
                # 特征比对
                try:
                    sim = distance(id_image_feature_numpy, query_image_feature[0], 0)
                    code = 0
                    message = "成功"
                    ainumber = "6"
                    similarity = float(as_num(sim * 100))
                    return return_compare_result(code, message, ainumber, similarity)
                except Exception as e:
                    code = 20000
                    message = str(e)
                    ainumber = "6"
                    similarity = None
                    return return_compare_result(code, message, ainumber, similarity)
            except Exception as e:
                code = 20000
                message = str(e)
                ainumber = "6"
                similarity = None
                return return_compare_result(code, message, ainumber, similarity)
        else:
            code = 20000
            message = "LICENSE过期"
            ainumber = "6"
            similarity = None
            return return_compare_result(code, message, ainumber, similarity)
    except Exception as e:
        code = 20000
        message = str(e)
        ainumber = "6"
        similarity = None
        if error_log_flag == True:
            logger_error.error("compareImageAndFeature - " + str(e))
        return return_compare_result(code, message, ainumber, similarity)

# 图片生成特征值
def feature_extract(image_base64):
    try:
        # is_invalid = is_license_valid(license_check)
        if is_invalid:
            try:
                if info_log_flag == True:
                    logger_info.info("getFeature")
                # 首先经过图像质量评估模块
                code, message, face_base64 = getFaceQuality(image_base64)
                # 代表code是1代表不经过图像质量评估
                if code == 1:
                    if (len(image_base64) == 0):
                        code = 10001
                        message = "入参base64为空"
                        ainumber = "6"
                        data = None
                        return return_feature_result(code, message, ainumber, data)
                    try:
                        image = base64_to_image(image_base64)
                    except Exception as e:
                        code = 10001
                        message = "入参base64异常"
                        ainumber = "6"
                        data = None
                        return return_feature_result(code, message, ainumber, data)
                    # 图像增强
                    image_augment = imageAugmentation(image)
                    # query人脸检测
                    image_face_detect = faceDetectRot(image_augment, face_detect_model_flag,
                                                            getFeatureErrRotFlag, angleRotFlag,
                                                            face_detect_model)
                    if image_face_detect is None:
                        code = 10060
                        message = "人脸检测失败"
                        ainumber = "6"
                        data = None
                        return return_feature_result(code, message, ainumber, data)
                # 代表code是0代表经过图像质量后都合规
                elif code == 0:
                    image_face_detect = np.transpose(base64_to_image(face_base64), (2, 0, 1))
                else:
                    ainumber = "6"
                    data = None
                    return return_feature_result(code, message, ainumber, data)
                # 提取特征
                try:
                    init_feature = face_recognition.getFeature(image_face_detect)
                    init_feature = preprocessing.normalize(init_feature)
                    feature_bytes = init_feature.tobytes()
                    feature_base64_bytes = base64.encodebytes(feature_bytes)
                    feature_base64_bytes_to_str = str(feature_base64_bytes.decode())
                    code = 0
                    message = "成功"
                    ainumber = "6"
                    data = {}
                    data['feature'] = feature_base64_bytes_to_str
                    data['version'] = faceEngineVersion
                    data['quality'] = "0.95"
                    return return_feature_result(code, message, ainumber, data)
                except Exception as e:
                    code = 10070
                    message = "抽取特征值失败"
                    ainumber = "6"
                    data = None
                    return return_feature_result(code, message, ainumber, data)
            except Exception as e:
                code = 20000
                message = str(e)
                ainumber = "6"
                data = None
                return return_feature_result(code, message, ainumber, data)
        else:
            code = 20000
            message = "LICENSE过期"
            ainumber = "6"
            data = None
            return return_feature_result(code, message, ainumber, data)
    except Exception as e:
        code = 20000
        message = str(e)
        ainumber = "6"
        data = None
        if error_log_flag == True:
            logger_error.error("getFeature - " + str(e))
        return return_feature_result(code, message, ainumber, data)

# 批量生成特征值
def batch_feature_extract(request_data):
    try:
        # 初始化data_results字典
        data_results = {
            "version": faceEngineVersion,
            "result": {
                "success": {
                    "total": 0,
                    "data": []
                },
                "failed": {
                    "total": 0,
                    "data": []
                }
            }
        }
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(handle_image, item) for item in request_data]
            for future in futures:
                result = future.result()
                if "feature" in result:
                    data_results["result"]["success"]["total"] += 1
                    data_results["result"]["success"]["data"].append(result)
                else:
                    data_results["result"]["failed"]["total"] += 1
                    data_results["result"]["failed"]["data"].append(result)
        code = 0
        message = "成功"
        ainumber = "6"
        return return_batch_feature_result(code, message, ainumber, data_results)
    except Exception as e:
        code = 20000
        message = str(e)
        ainumber = "6"
        data_null = None
        return return_batch_feature_result(code, message, ainumber, data_null)


# 特征比特征
def feature_and_feature_verify(feature_one, feature_two):
    try:
        # is_invalid = is_license_valid(license_check)
        if is_invalid:
            if info_log_flag == True:
                logger_info.info("compareFeatureAndFeature")
            if (len(feature_one) == 0 or len(feature_two) == 0):
                code = 10002
                message = "入参特征值为空"
                ainumber = "6"
                similarity = None
                return return_compare_result(code, message, ainumber, similarity)
            try:
                input_feature_1_base64_bytes = feature_one.encode()
                input_feature_1_bytes = base64.decodebytes(input_feature_1_base64_bytes)
                input_feature_1_numpy = np.frombuffer(input_feature_1_bytes, dtype=np.float32)
                input_feature_2_base64_bytes = feature_two.encode()
                input_feature_2_bytes = base64.decodebytes(input_feature_2_base64_bytes)
                input_feature_2_numpy = np.frombuffer(input_feature_2_bytes, dtype=np.float32)
            except Exception as e:
                code = 10002
                message = "入参特征值异常"
                ainumber = "6"
                similarity = None
                return return_compare_result(code, message, ainumber, similarity)
            try:
                sim = distance(input_feature_1_numpy, input_feature_2_numpy, 0)
                code = 0
                message = "成功"
                ainumber = "6"
                similarity = float(as_num(sim * 100))
                return return_compare_result(code, message, ainumber, similarity)
            except Exception as e:
                code = 20000
                message = str(e)
                ainumber = "6"
                similarity = None
                return return_compare_result(code, message, ainumber, similarity)
        else:
            code = 20000
            message = "LICENSE过期"
            ainumber = "6"
            similarity = None
            return return_compare_result(code, message, ainumber, similarity)
    except Exception as e:
        code = 20000
        message = str(e)
        ainumber = "6"
        similarity = None
        if error_log_flag == True:
            logger_error.error("compareFeatureAndFeature - " + str(e))
        return return_compare_result(code, message, ainumber, similarity)





