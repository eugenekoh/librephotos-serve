import cv2
import numpy as np
from flask import Flask, jsonify, request

import fer
import pic
import sic

app = Flask(__name__)


@app.route('/generate_pic', methods=['POST'])
def generate_pic():
    # get image
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return jsonify(pic.generate(img, request.form['personality']))


@app.route('/generate_sic', methods=['POST'])
def generate_sic():
    # get image
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return jsonify(sic.generate(img))


@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    # get image
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return jsonify(fer.predict_emotion(img))


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
