#coding=utf-8
import cv2
import dlib
import logging
import os
from cn.gtmap.utils import vector_distance_utils

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='face.log',
                filemode='w')

detector = dlib.get_frontal_face_detector()
base_path = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
shape_predictor = dlib.shape_predictor(base_path + '\\support\\dlib\\shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1(base_path + '\\support\\dlib\\dlib_face_recognition_resnet_model_v1.dat')
font=cv2.FONT_HERSHEY_SIMPLEX

idcard_img = cv2.imread(base_path + '\\support\\img\\id_pic.jpg')
cv2.imshow('ID Card Image', idcard_img)
idcard_face = detector(idcard_img, 1)
if (len(idcard_face) > 0):
    for k,d in enumerate(idcard_face):
        idcard_face_shape = shape_predictor(idcard_img, d)
        idcard_face_descriptor = face_recognition_model.compute_face_descriptor(idcard_img, idcard_face_shape)
else:
    logging.error('图片中无人脸！')
    exit()

video_capture = cv2.VideoCapture(0)
while True:
    ret,capture_img = video_capture.read()
    capture_faces = detector(capture_img, 1)
    if (len(capture_faces) > 0):
        for k,d in enumerate(capture_faces):
            cv2.rectangle(capture_img, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255))
            capture_face_shape = shape_predictor(capture_img, d)
            capture_face_descriptor = face_recognition_model.compute_face_descriptor(capture_img, capture_face_shape)
            face_euclidean_distance = vector_distance_utils.cal_euclidean_distance(idcard_face_descriptor, capture_face_descriptor)
            if face_euclidean_distance < 0.45:
                cv2.putText(capture_img, 'matched', (0, 40), font, 1.2, (0, 0, 255), 2)
                logging.debug("matched,disance:"+str(face_euclidean_distance))
            else:
                cv2.putText(capture_img, 'dismatched', (0, 40), font, 1.2, (255, 255, 255), 2)
                logging.debug("dismatched,disance:"+str(face_euclidean_distance))

    cv2.imshow('Face Verification', capture_img)
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()