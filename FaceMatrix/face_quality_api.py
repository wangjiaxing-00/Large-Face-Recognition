import requests, json
import myJsonEncoder
import math
import base64
import os
import configparser

# 读取配置参数
config = configparser.ConfigParser()
config.read("./config/algorithmPara.ini", "utf-8")
face_quality_url = config.get("algorithm", "face_quality_url")
# 人脸图像质量评估模块开关
face_quality_switch = config.getboolean('algorithm', 'face_quality_switch')
brightness_min_threshold = config.getfloat('algorithm', 'brightness_min_threshold')
brightness_max_threshold = config.getfloat('algorithm', 'brightness_max_threshold')
definition_threshold = config.getfloat('algorithm', 'definition_threshold')
face_head_angle_yaw_threshold = config.getfloat('algorithm', 'face_head_angle_yaw_threshold')
face_head_angle_pitch_threshold = config.getfloat('algorithm', 'face_head_angle_pitch_threshold')
face_head_angle_roll_threshold = config.getfloat('algorithm', 'face_head_angle_roll_threshold')
face_quality_threshold = config.getfloat('algorithm', 'face_quality_threshold')
face_mask_threshold = config.get('algorithm', 'face_mask_threshold')

def getFaceQuality(image_base64):
    try:
        # 如果图像质量评估模块打开
        if face_quality_switch == True:
            post_json = json.dumps({
                "query_image_base64": image_base64
            }, cls=myJsonEncoder.MyEncoder)
            requests_post = requests.post(face_quality_url, post_json,
                                 headers={'Content-Type': 'application/json', 'Connection': 'close'})
            if requests_post.status_code == requests.codes.ok:
                result = requests_post.json()
                if result["code"] == 0:
                    code = result["code"]
                    message = result["message"]
                    face_base64 = result["face_base64"]
                    # 独立判断，不要嵌套
                    # 判断口罩，若口罩不为空
                    if result["face_mask"] is not None:
                        if face_mask_threshold == "1":
                            if result["face_mask"] == "1":
                                code = 10040
                                message = "佩戴口罩不合规"
                                face_base64 = None
                                return code, message, face_base64
                    # 判断角度,若角度不为空
                    if result["face_angle"] is not None:
                        if abs(float(result["face_angle"]["yaw"])) > face_head_angle_yaw_threshold:
                            code = 10041
                            message = "人脸角度yaw不合规,yaw:" + result["face_angle"]["yaw"]
                            face_base64 = None
                            return code, message, face_base64
                        if abs(float(result["face_angle"]["pitch"])) > face_head_angle_pitch_threshold:
                            code = 10042
                            message = "人脸角度pitch不合规,pitch:" + result["face_angle"]["pitch"]
                            face_base64 = None
                            return code, message, face_base64
                        if abs(float(result["face_angle"]["roll"])) > face_head_angle_roll_threshold:
                            code = 10043
                            message = "人脸角度roll不合规,roll:" + result["face_angle"]["roll"]
                            face_base64 = None
                            return code, message, face_base64
                    # 判断光照,若光照不为空
                    if result["face_brightness"] is not None:
                        if float(result["face_brightness"]) < brightness_min_threshold:
                            code = 10044
                            message = "光照不合规,光照:" + result["face_brightness"]
                            face_base64 = None
                            return code, message, face_base64
                        elif float(result["face_brightness"]) > brightness_max_threshold:
                            code = 10044
                            message = "光照不合规,光照:" + result["face_brightness"]
                            face_base64 = None
                            return code, message, face_base64
                    # 清晰度,若清晰度不为空
                    if result["face_definition"] is not None:
                        if float(result["face_definition"]) < definition_threshold:
                            code = 10045
                            message = "清晰度不合规,清晰度:" + result["face_definition"]
                            face_base64 = None
                            return code, message, face_base64
                    # 总质量,若总质量不为空
                    if result["face_quality"] is not None:
                        if float(result["face_quality"]) < face_quality_threshold:
                            code = 10046
                            message = "总质量不合规,总质量:" + result["face_definition"]
                            face_base64 = None
                            return code, message, face_base64
                    return code, message, face_base64
                else:
                    code = result["code"]
                    message = result["message"]
                    face_base64 = None
                    return code, message, face_base64
            else:
                code = 20000
                message = "人脸图像质量评估接口网络异常"
                face_base64 = None
                return code, message, face_base64
        # 如果图像质量评估模块关闭
        else:
            code = 1
            message = "图像质量评估模块关闭"
            face_base64 = image_base64
            return code, message, face_base64
    except Exception as e:
        code = 20000
        message = "人脸图像质量评估接口异常，" + str(e)
        face_base64 = None
        return code, message, face_base64







