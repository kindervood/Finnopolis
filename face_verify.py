from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os


from deepface import DeepFace

choice_backend = 0
choice_metric = 2
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface',
            'mediapipe']  # opencv или ssd - скорость \ retinaface и mtcnn - качество
metrics = ["cosine", "euclidean", "euclidean_l2"]  # методы сравнения

def face_analyze(img1):
    try:
        res_attributes = DeepFace.analyze(img_path=img1, actions=("age", "emotion"), detector_backend=backends[0])

        return res_attributes

    except Exception as _ex:
        return _ex


def get_similar_faces(img1, img2):
    try:
        # результат сравнения двух лиц
        res_compare = DeepFace.verify(img1_path=img1, img2_path=img2, distance_metric=metrics[choice_metric],
                                      detector_backend=backends[choice_backend])
        return res_compare
    except Exception as _ex:
        return _ex


def find_face_in_base(img1):
    # распознование лица из базы данных \ ищет похожие
    df = DeepFace.find(img_path=img1, db_path="images",
                       detector_backend=backends[choice_backend]).identity
    res_list_recognition = []
    for i in range(len(df)):
        res_list_recognition.append(df[i])
    return res_list_recognition

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)