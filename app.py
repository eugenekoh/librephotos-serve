import pic
import fer
from flask import Flask, jsonify, request
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    # get image
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return jsonify(pic.generate(img, request.form['personality']))

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    # get image
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    return jsonify(fer.predict_emotion(img))

if __name__ == "__main__":
    app.run()
