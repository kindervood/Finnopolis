import warnings
from face_verify import *
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import requests
from flask import Flask, jsonify, request, make_response
import argparse

warnings.filterwarnings("ignore")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

@app.route('/analyze')
def analyze():
    body = request.get_json()
    res = face_analyze(body['address1'])

    return jsonify(res)


@app.route('/find')
def find():
    body = request.get_json()
    res = find_face_in_base(body['address1'])

    return jsonify(res)


@app.route('/getsimilar')
def get_similar():
    body = request.get_json()
    res = get_similar_faces(body['address1'], body['address2'])

    return jsonify(res)


@app.route('/check')
def check():
    body = request.get_json()

    frame = cv2.imread(body['address1'])
    frame = imutils.resize(frame, width=400)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

    #Put Text and rect over the frame
    #cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    #cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    return label

if __name__ == '__main__':
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    maskNet = load_model("mask_detector.model")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=5000,
        help='Port of serving api')
    args = parser.parse_args()
    app.run(host='127.0.0.1', port=args.port)
