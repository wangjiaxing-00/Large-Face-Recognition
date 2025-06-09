import cv2
import numpy as np
import rflm106det
from face_detect.insightface.app import FaceAnalysis
from face_detect.insightface.data import get_image as ins_get_image

if __name__ == '__main__':
    # detection,landmark_3d_68,landmark_2d_106
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640,640))
    img = ins_get_image('t1')
    faces = app.get(img)
    tim = img.copy()
    color =(200,160,75)
    for face in faces:
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(int) # 转换为整型
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            cv2.circle(tim, p, 1, color, cv2.LINE_AA)
        cv2.imwrite('./test_out.jpg', tim)