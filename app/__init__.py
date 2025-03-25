# app/__init__.py
from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    # 允许跨域请求
    CORS(app, supports_credentials=True)  # 允许跨域请求时携带凭证

    # 注册蓝图（模块化路由）
    from app.routes.scan import scan_bp
    from app.routes.yolo import yolo_bp
    from app.routes.correct import correct_bp
    from app.routes.deblur import deblur_bp

    app.register_blueprint(scan_bp)
    app.register_blueprint(yolo_bp)
    app.register_blueprint(correct_bp)
    app.register_blueprint(deblur_bp)

    return app
