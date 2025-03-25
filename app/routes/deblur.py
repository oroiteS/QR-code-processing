import os
import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from DeblurGANv2.predict import main, Predictor

deblur_bp = Blueprint('deblur', __name__)


def get_project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@deblur_bp.route('/deblur', methods=['POST'])
def deblur():
    if 'image' not in request.files:
        return jsonify({'error': '没有找到图片文件'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '未选择图片'}), 400

    try:
        # 直接读取上传的文件内容
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': '无法读取图片内容'}), 400

        # 转换颜色空间
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 创建输出目录
        project_root = get_project_root()
        submit_dir = os.path.join(project_root, 'statc', 'submit')
        os.makedirs(submit_dir, exist_ok=True)

        # 直接使用 Predictor 处理图像
        predictor = Predictor(weights_path=os.path.join(project_root, 'DeblurGANv2', 'weights', 'fpn_mobilenet.h5'))
        pred = predictor(img, None)

        # 保存处理后的图片
        output_path = os.path.join(submit_dir, 'temp.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

        # 读取处理后的图片并转换为字节
        with open(output_path, 'rb') as f:
            processed_image_bytes = f.read()

        # 清理输出文件
        if os.path.exists(output_path):
            os.remove(output_path)

        return jsonify({
            'image': processed_image_bytes.hex()
        })

    except Exception as e:
        print(f"去模糊处理错误: {str(e)}")
        return jsonify({'error': f'去模糊处理失败: {str(e)}'}), 500
