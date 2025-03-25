# app/routes/auth.py
import cv2
import numpy as np
from flask import Blueprint, request, jsonify

scan_bp = Blueprint('scan', __name__)


@scan_bp.route('/scan', methods=['POST'])
def scan():
    # 检查是否有文件被上传
    if 'image' not in request.files:
        return jsonify({'error': '没有找到图片文件'}), 400

    file = request.files['image']

    # 检查文件名是否为空
    if file.filename == '':
        return jsonify({'error': '未选择图片'}), 400

    # 读取图片文件的字节数据
    file_bytes = file.read()

    # 将字节数据转换为numpy数组
    nparr = np.frombuffer(file_bytes, np.uint8)

    # 使用cv2解码图片
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': '图片解码失败'}), 400

    model = cv2.wechat_qrcode.WeChatQRCode()
    qr_code = model.detectAndDecode(image)
    if len(qr_code) > 0 and len(qr_code[0]) > 0:
        url = qr_code[0][0]
        return jsonify({'message': f'图片上传成功：{url}'}), 200
    else:
        return jsonify({'message': '图片扫描失败'}), 400
