"""Flask calculator and image inference service. No model runs on import."""
import argparse
import hmac
import io
import math
import os
import threading
import warnings
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from calculator import calculate


def _to_number(raw):
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
        raise ValueError('操作数必须是数字')
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            raise ValueError('操作数不能为空')
        try:
            raw = int(raw)
        except ValueError:
            raw = float(raw)
    try:
        if not math.isfinite(raw):
            raise ValueError('操作数必须是有限数字')
    except OverflowError as error:
        raise ValueError('操作数超出支持范围') from error
    return raw


def create_app(predictor=None, api_key=None):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
    key = os.environ.get('API_KEY', '') if api_key is None else api_key
    lock = threading.Lock()

    @app.before_request
    def authenticate():
        if request.endpoint not in ('health', 'index') and key:
            supplied = request.headers.get('X-API-Key', '')
            if not hmac.compare_digest(supplied.encode(), key.encode()):
                return jsonify(error='无效 API key'), 401

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def unexpected_error(error):
        app.logger.exception('request failed')
        return jsonify(error='服务器处理失败'), 500

    @app.get('/')
    def index():
        return jsonify(service='ML recruitment API', endpoints=['/calculate', '/predict', '/health'])

    @app.get('/health')
    def health():
        return jsonify(status='ok', model_ready=predictor is not None)

    @app.post('/')
    @app.post('/calculate')
    def handle_calculate():
        if request.is_json:
            body = request.get_json()
            if not isinstance(body, dict):
                return jsonify(error='JSON 必须是对象'), 400
        elif request.mimetype in ('multipart/form-data', 'application/x-www-form-urlencoded'):
            body = request.form
            if any(len(body.getlist(k)) > 1 for k in ('Operation', 'Data_A', 'Data_B')):
                return jsonify(error='字段不得重复'), 400
        else:
            return jsonify(error='使用 JSON 或表单请求体'), 415
        if any(k not in body for k in ('Operation', 'Data_A', 'Data_B')):
            return jsonify(error='缺少 Operation、Data_A 或 Data_B'), 400
        try:
            operation = body['Operation']
            if not isinstance(operation, str):
                raise ValueError('Operation 必须是字符串')
            result = calculate(operation.strip(), _to_number(body['Data_A']), _to_number(body['Data_B']))
        except (ValueError, TypeError, OverflowError) as error:
            return jsonify(error=str(error)), 400
        return jsonify(Result=result)

    @app.post('/predict')
    def predict():
        if request.mimetype != 'multipart/form-data':
            return jsonify(error='file 必须通过 multipart/form-data 上传'), 415
        uploaded = request.files.get('file')
        if uploaded is None or not uploaded.filename:
            return jsonify(error='缺少 file 文件字段'), 400
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                image = Image.open(io.BytesIO(uploaded.read()))
                if image.width * image.height > 4_000_000:
                    return jsonify(error='图片不得超过 400 万像素'), 400
                if image.format not in ('PNG', 'JPEG', 'WEBP'):
                    return jsonify(error='仅支持 PNG/JPEG/WEBP'), 400
                image.load()
                image = image.convert('RGB')
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            return jsonify(error='文件不是可解码的有效图片'), 400
        if predictor is None:
            return jsonify(error='尚未配置推理模型'), 503
        # Serialize access for one-process GPU inference; multiple workers replicate GPU weights.
        with lock:
            result = predictor.predict(image)
        return jsonify(result)

    return app


def application():
    kind = os.environ.get('MODEL_KIND', 'none')
    device = os.environ.get('MODEL_DEVICE', 'auto')
    predictor = None
    if kind == 'classifier':
        from model_service import Classifier
        predictor = Classifier(device)
    elif kind == 'derainer':
        from model_service import Derainer
        predictor = Derainer(os.environ['DERAIN_CHECKPOINT'], device)
    elif kind != 'none':
        raise ValueError('MODEL_KIND must be none/classifier/derainer')
    return create_app(predictor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--model', choices=['none', 'classifier', 'derainer'], default='none')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    os.environ['MODEL_KIND'], os.environ['MODEL_DEVICE'] = args.model, args.device
    if args.checkpoint:
        os.environ['DERAIN_CHECKPOINT'] = args.checkpoint
    application().run(host=args.host, port=args.port, debug=False, use_reloader=False)
