from flask import Flask, request, render_template, send_from_directory, url_for
import os
from werkzeug.utils import secure_filename
from yolo_infer import process_video,process_image  # 你的视频处理逻辑

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/'

# 创建目录
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)  # 安全命名
        input_path = os.path.join(UPLOAD_FOLDER, filename)

        f.save(input_path)
        
        # 获取前端传来的类别，可能是多个，用list接收
        selected_classes = request.form.getlist('classes')
        # 转为int列表
        classes = list(map(int, selected_classes))

        suff=filename.split('.')[-1]

        output_filename = os.path.splitext(filename)[0] + f'_output.{suff}'
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        print(output_path)

        if suff in {'mp4','mov','avi'}:
            process_video(input_path, output_path, classes)    
        else:
            process_image(input_path, output_path, classes)
        
        video_url = url_for('static', filename=f'{output_filename}')
        download_url = url_for('download_file', filename=output_filename)
        print(video_url,download_url)
        return render_template('result.html', video_path=video_url, download_url=download_url)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
