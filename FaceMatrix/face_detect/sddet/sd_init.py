from face_detect.sddet.facedetectsd import SD
import onnxruntime as ort

def scrfd_model_init(model_file):
    detector = SD(model_file)
    return detector

def keypoint_model_init(key_list):
    kp_detector = ort.InferenceSession(key_list[0],providers=key_list[1])
    return kp_detector