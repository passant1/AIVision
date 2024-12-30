import os
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import subprocess
import shutil
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:8080", "http://localhost:8081"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept"]
    }
})

# 配置上传文件和风格图片路径
UPLOAD_FOLDER = 'uploads'
STYLES_FOLDER = 'styles'  # 添加风格图片目录
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STYLES_FOLDER'] = STYLES_FOLDER

# 确保目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STYLES_FOLDER, exist_ok=True)

# 风格图片映射
STYLE_IMAGES = {
    'vangogh': 'starry.jpg',      # 星夜
    'sunflower': 'sunflower.jpg'  # 向日葵
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_file():
    print("收到上传请求")
    if 'file' not in request.files:
        print("未找到文件")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    style = request.form.get('style', 'vangogh')
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # 保存上传的内容图片
            content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
            file.save(content_path)
            
            # 先返回上传成功响应
            return jsonify({
                'message': 'File uploaded successfully',
                'url': '/api/image/content'
            })
            
        except Exception as e:
            print(f"上传错误: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

# 添加新的处理接口
@app.route('/api/process', methods=['POST'])
def process_image():
    try:
        style = request.json.get('style', 'vangogh')
        
        # 准备文件路径
        content_path = os.path.join(app.config['UPLOAD_FOLDER'], 'content.jpg')
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output.jpg')
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], 'style.jpg')
        
        # 复制风格图片
        style_image = STYLE_IMAGES.get(style, 'starry.jpg')
        style_source = os.path.join(app.config['STYLES_FOLDER'], style_image)
        shutil.copy(style_source, style_path)
        
        # 运行风格迁移脚本
        process = subprocess.Popen(
            ['python', 'style_transfer.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 实时读取进度
        while True:
            line = process.stdout.readline()
            if not line:
                break
            if line.startswith('PROGRESS:'):
                progress = float(line.split(':')[1])
                print(f"Processing progress: {progress}%")
        
        process.wait()
        
        if process.returncode == 0:
            return jsonify({
                'message': 'Processing completed',
                'url': '/api/image/output'
            })
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        print(f"处理错误: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/image/<filename>')
def get_image(filename):
    """获取处理后的图片"""
    if filename not in ['content', 'output']:
        return jsonify({'error': 'Invalid filename'}), 400
        
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{filename}.jpg')
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Image not found'}), 404
        
    return send_file(file_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
