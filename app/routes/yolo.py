import cv2 as cv
import numpy as np
from flask import Blueprint, request, jsonify

from ultralytics import YOLOv10

yolo_bp = Blueprint('yolo', __name__)

# 初始化 YOLOv10 模型
weights = './yolov10/runs/detect/train/weights/best.pt'
model = YOLOv10(weights)


@yolo_bp.route('/detect', methods=['POST'])
def detect():

    try:
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
        image = cv.imdecode(nparr, cv.IMREAD_COLOR)

        if image is None:
            return jsonify({'error': '图片解码失败'}), 400

        # 使用YOLOv10进行目标检测
        results = model.predict(image)

        detected_objects = []
        processed_image = image.copy()
        cropped_image = processed_image

        for r in results:
            boxes = r.boxes.xywh
            for box in boxes:
                x, y, w, h = box
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)

                # 在图像上绘制边界框
                cv.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 提取检测到的对象
                cropped_image = image[y1:y2, x1:x2]

                # 将裁剪的图像调整为统一大小（可选）
                cropped_image = cv.resize(cropped_image, (640, 640))

                # 将检测到的对象信息添加到列表中
                detected_objects.append({
                    'coordinates': {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    },
                    'confidence': float(r.boxes.conf[0])
                })

        # 将处理后的图像编码为JPEG格式
        _, buffer = cv.imencode('.jpg', cropped_image)
        image_bytes = buffer.tobytes()

        # 返回处理后的图像和检测结果
        return jsonify({
            'detected_objects': detected_objects,
            'image': image_bytes.hex()  # 将图像数据转换为十六进制字符串
        })

    except Exception as e:
        print(f"目标检测错误: {str(e)}")
        return jsonify({'error': '目标检测失败'}), 500
