import cv2 as cv
import numpy as np
from flask import Blueprint, request, jsonify

correct_bp = Blueprint('correct', __name__)


@correct_bp.route('/correct', methods=['POST'])
def correct():
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
        scr_img = cv.imdecode(nparr, cv.IMREAD_COLOR)

        if scr_img is None:
            return jsonify({'error': '图片解码失败'}), 400

        # 将图像调整为500*500
        scr_img = cv.resize(scr_img, (500, 500))

        # 亮度与对比度调节
        # np.zeros的作用是创建一个与scr_img大小相同的全黑图像
        contrast_img = np.zeros(scr_img.shape, scr_img.dtype)
        alpha = 1.8  # 对比度因子
        beta = -30  # 亮度因子
        for y in range(scr_img.shape[0]):  # 遍历图像的每一行和每一列
            for x in range(scr_img.shape[1]):
                for c in range(3):  # 因为是彩色图像，所以有三个通道
                    # np.clip的作用是在将原像素*alpha提高对比度并降低亮度之后确保其值在0-255之间
                    contrast_img[y, x, c] = np.clip(alpha * scr_img[y, x, c] + beta, 0, 255)
        # cv.imshow('contrast_img', contrast_img)
        # cv.waitKey(0)

        # 将contrast_img转换为灰度图
        gray_image = cv.cvtColor(contrast_img, cv.COLOR_BGR2GRAY)

        # 对灰度图进行双边滤波
        filter_image = cv.bilateralFilter(gray_image, 13, 26, 6)
        # 如果想使用中值滤波
        # filter_image = cv.medianBlur(gray_image, 3)

        # 对滤波后的图像进行反二值化处理
        _, binary_image = cv.threshold(filter_image, 210, 255, cv.THRESH_BINARY_INV)

        # 将反二值化后的图片再次反二值化
        bin_image = cv.bitwise_not(binary_image)

        # 创建一个核，这里使用3x3的矩形核
        kernel = np.ones((3, 3), np.uint8)
        # 对二值化后的图像进行腐蚀操作，迭代2次
        erode_image = cv.erode(binary_image, kernel, iterations=2)

        # 对腐蚀后的图像进行膨胀操作，迭代19次
        dilate_image = cv.dilate(erode_image, kernel, iterations=19)

        # 使用Canny边缘检测
        canny_image = cv.Canny(dilate_image, 10, 100, apertureSize=3, L2gradient=False)

        # 克隆图像，用于绘制直线
        all_lines_image = canny_image.copy()
        # 使用Hough变换检测直线
        rho_res = 5  # rho的精度
        theta_res = np.pi / 180  # theta的精度
        threshold = 100  # 阈值
        # cv.HoughLines的作用是在图像中寻找直线。它使用Hough变换来检测图像中的直线
        lines = cv.HoughLines(all_lines_image, rho_res, theta_res, threshold)
        # 如果检测到直线，lines将不是空的
        if lines is not None:
            for line in lines[:, 0]:
                rho, theta = line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
                # 在图像上绘制直线
                cv.line(all_lines_image, pt1, pt2, (255, 255, 255), 1, cv.LINE_AA)  # 使用BGR格式的白色

        # 计算二维码的四个顶点坐标
        A = 50.0
        B = np.pi / 180 * 20  # 20度
        resLines = lines[:]  # 复制lines列表
        removeIndex = set()
        countLess4 = 0
        countMore4 = 0
        
        # 添加最大迭代次数限制，防止死循环
        max_iterations = 500
        current_iterations = 0

        while True:
            for i in range(len(resLines)):
                for j in range(i + 1, len(resLines)):
                    rho1, theta1 = resLines[i][0]
                    rho2, theta2 = resLines[j][0]

                    # theta大于pi，减去进行统一
                    if theta1 > np.pi:
                        theta1 -= np.pi
                    if theta2 > np.pi:
                        theta2 -= np.pi

                    # 记录需要删除的lines
                    thetaFlag = abs(theta1 - theta2) <= B or \
                                (theta1 > np.pi / 2 > theta2 and np.pi - theta1 + theta2 < B) or \
                                (theta2 > np.pi / 2 > theta1 and np.pi - theta2 + theta1 < B)

                    if abs(abs(rho1) - abs(rho2)) <= A and thetaFlag:
                        removeIndex.add(j)

            # 删除多余的lines
            res = [resLines[i] for i in range(len(resLines)) if i not in removeIndex]
            resLines = res[:]
            
            # 递增迭代计数器
            current_iterations += 1

            # 直到删除只剩4条直线或达到最大迭代次数
            if len(resLines) > 4:
                A += 4
                B += 2 * np.pi / 180
                countMore4 += 1
                if countMore4 % 50 == 0:
                    print("countMore4：", countMore4)
            elif len(resLines) < 4:
                B -= np.pi / 180
                countLess4 += 1
                if countLess4 % 50 == 0:
                    print("countLess4：", countLess4)
            else:
                print("删除后的剩余直线个数：", len(resLines))
                break
                
            # 检查是否达到最大迭代次数，如果是则跳出循环
            if current_iterations >= max_iterations:
                print(f"达到最大迭代次数 {max_iterations}，当前剩余直线个数: {len(resLines)}")
                break
                
        # 检查是否有足够的直线继续处理
        if len(resLines) < 4:
            return jsonify({'error': '无法检测到足够的直线来校正图像'}), 400

        # 在canny图上画出剩下的4条拟合直线
        four_lines_image = canny_image.copy()
        four_lines = resLines[:]
        for line in four_lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv.line(four_lines_image, pt1, pt2, (255, 255, 255), 1)

        # 求出四条定位直线在图像界内的四个交点。
        threshold = 0.2 * min(canny_image.shape[0], canny_image.shape[1])
        points = []
        for i in range(len(four_lines)):
            for j in range(i + 1, len(four_lines)):
                rho1, theta1 = four_lines[i][0]
                rho2, theta2 = four_lines[j][0]
                if theta1 == 0:
                    theta1 = 0.01
                if theta2 == 0:
                    theta2 = 0.01
                a1 = np.cos(theta1)
                b1 = np.sin(theta1)
                a2 = np.cos(theta2)
                b2 = np.sin(theta2)
                denominator = a2 * b1 - a1 * b2
                if denominator != 0:  # 检查分母是否为零
                    try:
                        x = (rho2 * b1 - rho1 * b2) / denominator
                        y = (rho1 - a1 * x) / b1
                        pt = (int(np.round(x)), int(np.round(y)))
                        if (canny_image.shape[1] + threshold >= pt[0] >= 0 - threshold and
                                canny_image.shape[0] + threshold >= pt[1] >= 0 - threshold):
                            points.append(pt)
                    except OverflowError:
                        continue
                else:
                    # 如果分母为零，说明直线平行，跳过
                    continue

        # 将获取到的交点按顺时针排序并保存在sortPoints中
        sort_points = []
        min_x = float('inf')
        index = -1
        for i, pt in enumerate(points):
            if min_x > pt[0]:
                min_x = pt[0]
                index = i
        left = points[index]
        points.remove(left)
        sort_points.append(left)
        while points:
            min_grad = float('inf')
            idx = -1
            for i, pt in enumerate(points):
                grad = (pt[1] - left[1]) / (pt[0] - left[0]) if (pt[0] - left[0]) != 0 else float('inf')
                if min_grad > grad:
                    min_grad = grad
                    idx = i
            sort_points.append(points[idx])
            left = points[idx]
            points.remove(points[idx])

        # 按照sortPoints四个点进行仿射变换
        # 得到最小的边长
        min_side = min(scr_img.shape[0], scr_img.shape[1])
        # 得到图像的中心点
        center_x = scr_img.shape[0] // 2
        center_y = scr_img.shape[1] // 2
        # 定义一个源四边形和一个目的四边形
        src_qua = np.array(sort_points, dtype=np.float32)
        dst_qua = np.array([[center_x - 0.45 * min_side, center_y - 0.45 * min_side],
                            [center_x + 0.45 * min_side, center_y - 0.45 * min_side],
                            [center_x + 0.45 * min_side, center_y + 0.45 * min_side],
                            [center_x - 0.45 * min_side, center_y + 0.45 * min_side]], dtype=np.float32)
        # 得到一个从源四边形到目的四边形的透视变换矩阵
        print(src_qua, dst_qua)
        trans_mtx = cv.getPerspectiveTransform(src_qua, dst_qua)
        # 仿射变换
        warped_image = cv.warpPerspective(bin_image, trans_mtx, (canny_image.shape[1], canny_image.shape[0]))
        if warped_image.ndim == 2:  # 灰度图只有一个通道
            warped_image = cv.cvtColor(warped_image, cv.COLOR_GRAY2BGR)

        # 将校正后的图像编码为JPEG格式
        _, buffer = cv.imencode('.jpg', warped_image)
        # 将图像数据转换为字节
        image_bytes = buffer.tobytes()

        # 创建响应对象并设置图像数据
        from flask import send_file
        import io
        return send_file(
            io.BytesIO(image_bytes),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name='corrected_image.jpg'
        )

    except Exception as e:
        # 捕获所有可能的异常
        print(f"图像处理错误: {str(e)}")
        return jsonify({'error': '图像处理失败'}), 500

