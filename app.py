from flask import Flask, request, render_template, send_from_directory
import os
from yolo_infer import process_video

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        input_path = os.path.join(UPLOAD_FOLDER, f.filename)
        output_path = os.path.join(OUTPUT_FOLDER, 'output.mp4')
        f.save(input_path)
        process_video(input_path, output_path)
        return render_template('result.html', video_path='static/output.mp4')
    return render_template('index.html')

@app.route('/download')
def download_file():
    return send_from_directory(OUTPUT_FOLDER, 'output.mp4', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
