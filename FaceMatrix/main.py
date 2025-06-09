import os
from flask import Flask, request, jsonify, Response
import json
from app_run import return_compare_result, return_feature_result, return_batch_feature_result
from app_run import image_and_image_verify, image_and_feature_verify, feature_extract, batch_feature_extract, feature_and_feature_verify


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/')
def hello_world():
    return 'Hello World!'

# 接口1：图片比图片
@app.route('/algorithm/compareImageAndImage', methods=['POST'])
def compare_image_and_image():
    try:
        query_image_base64 = request.json['query_image_base64']
        id_image_base64 = request.json['id_image_base64']
        return_verify_result = image_and_image_verify(query_image_base64, id_image_base64)
        return return_verify_result
    except Exception as e:
        code = 20000
        message = "请求字段错误"
        ainumber = None
        similarity = None
        return return_compare_result(code, message, ainumber, similarity)


# 接口2：图片比特征
@app.route('/algorithm/compareImageAndFeature', methods=['POST'])
def compare_image_and_feature():
    try:
        query_image_base64 = request.json['query_image_base64']
        id_image_feature = request.json['id_image_feature']
        return_verify_result = image_and_feature_verify(query_image_base64, id_image_feature)
        return return_verify_result
    except Exception as e:
        code = 20000
        message = "请求字段错误"
        ainumber = None
        similarity = None
        return return_compare_result(code, message, ainumber, similarity)

# 接口3：图片生成特征值
@app.route('/algorithm/getFeature', methods=['POST'])
def get_feature():
    try:
        image_base64 = request.json['imageBase64']
        return_feature_result = feature_extract(image_base64)
        return return_feature_result
    except Exception as e:
        code = 20000
        message = "请求字段错误"
        ainumber = None
        data = None
        return return_feature_result(code, message, ainumber, data)

# 接口4：批量图片生成特征值
@app.route('/algorithm/getBatchFeature', methods=['POST'])
def get_batch_feature():
    try:
        data = request.json['data']
        return_batch_feature_result = batch_feature_extract(data)
        return return_batch_feature_result
    except Exception as e:
        code = 20000
        message = "请求字段错误"
        ainumber = None
        data = None
        return return_batch_feature_result(code, message, ainumber, data)

# 接口5：特征值比特征值
@app.route('/algorithm/compareFeatureAndFeature', methods=['POST'])
def compare_feature_and_feature():
    try:
        feature_one = request.json['featureOne']
        feature_two = request.json['featureTwo']
        return_verify_result = feature_and_feature_verify(feature_one, feature_two)
        return return_verify_result
    except Exception as e:
        code = 20000
        message = "请求字段错误"
        ainumber = None
        similarity = None
        return return_compare_result(code, message, ainumber, similarity)


if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=6001, threaded=False, debug=False)
    app.run()
