from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2

app = Flask(__name__)

@app.route('/')
def index():
    # Ensure test.html is placed in the templates/ folder
    return render_template('test.html')

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
        H_list = [[1,0,0],[0,1,0],[0,0,1]]

    return jsonify({'homographyMatrix': H_list})

if __name__ == '__main__':
    app.run(debug=True)
