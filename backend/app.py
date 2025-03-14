from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import requests
import base64
import os

app = Flask(__name__)

# Roboflow API Configuration
ROBOFLOW_API_KEY = "1uPusGf1GlueyunHYYhR"
ROBOFLOW_MODEL = "ballraitraining/1"
ROBOFLOW_VERSION = "1"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/uploader')
def uploader():
    return render_template('upload.html')

@app.route('/calculate_homography', methods=['POST'])
def calculate_homography():
    """
    Expects JSON with:
      {
        "pointsImage": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "pointsCourt": [[X1, Y1], [X2, Y2], [X3, Y3], [X4, Y4]]
      }
    Returns the 3x3 homography matrix as JSON.
    """
    data = request.get_json()
    points_image = np.array(data['pointsImage'], dtype='float32')
    points_court = np.array(data['pointsCourt'], dtype='float32')

    # Calculate the Homography matrix using RANSAC
    H, _ = cv2.findHomography(points_image, points_court, cv2.RANSAC, 5.0)

    # Convert NumPy array to list so it can be returned as JSON
    if H is not None:
        H_list = H.tolist()
    else:
        # If something went wrong, return an identity matrix or error
        H_list = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    return jsonify({'homographyMatrix': H_list})

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Receives a video frame (base64 image) and analyzes it using the Roboflow model.
    Returns the detected defense classification.
    """
    data = request.json
    image_data = data['image'].split(",")[1]  # Remove header from base64
    image_path = "temp.png"

    # Decode and save the image
    with open(image_path, "wb") as f:
        f.write(base64.b64decode(image_data))

    # Send image to Roboflow
    response = requests.post(
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}",
        files={"file": open(image_path, "rb")}
    )

    predictions = response.json()
    defense_prediction = predictions.get('predictions', [])

    # Extract the top label if available
    if defense_prediction:
        defense_label = defense_prediction[0]['class']
    else:
        defense_label = "Unknown Defense"

    # Clean up temporary image
    os.remove(image_path)

    return jsonify({"prediction": defense_label})

if __name__ == '__main__':
    app.run(debug=True)
