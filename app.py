from flask import Flask, request, render_template, redirect, url_for
import subprocess
import os

app = Flask(__name__)

# Path to store uploaded files
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/instructions')
def instructions():
    return render_template('instructions.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return 'No file part', 400
    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        # Save the uploaded file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Call video_analysis.py using subprocess and pass the file path
        try:
            subprocess.run(['python3', 'video_analysis.py', filepath], check=True)
            return 'Video processed successfully!'
        except subprocess.CalledProcessError as e:
            return f'Error during video analysis: {str(e)}', 500

    return redirect(url_for('upload'))

if __name__ == '__main__':
    app.run(debug=True)


