import os
from flask import Flask, request, jsonify
import base64
import numpy as np
import time
import configparser
import logging
import datetime
from flask import Response
import json
from UUIDTool import SnowflakeIDGenerator
import re
import myJsonEncoder
import requests, json
import concurrent.futures
import threading
from pymilvus import MilvusClient, DataType

# 配置
config = configparser.ConfigParser()
config.read("config/algorithmPara.ini", "utf-8")
feature_extract_url = config.get('http', 'feature_extract_url')


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # False返回json不按照字母顺序

# 连接服务，初始化 MilvusClient
client = MilvusClient(
     uri="http://127.0.0.1:19530",
     db_name="face"
)

# 获取集合的状态
res_status = client.get_load_state(
    collection_name="face_vector"
)
# 加载集合
# 枚举类型：LoadState.NotExist = 0
#         LoadState.NotLoad = 1
#         LoadState.Loading = 2
#         LoadState.Loaded = 3
if res_status["state"] == 1:
    client.load_collection(
         collection_name="face_vector"
    )


# 距离转换为分数 alpha=-4.83702,beta=5.56252
def distcance_to_score(dist, alpha=-9.81675, beta=10.93322):
    score = 1 - (1. / (1 + np.exp(alpha * dist + beta)))
    return 100*score


# 特征字符串类型转换为向量类型
def featureStringToVector(feature):
    feature_base64_bytes = feature.encode()
    feature_bytes = base64.decodebytes(feature_base64_bytes)
    feature_numpy = np.frombuffer(feature_bytes, dtype=np.float32)
    return feature_numpy

# 特征列表类型转换为特征字符串类型
def featureListToString(feature_list):
    # list类型转换为ndarray类型
    feature_ndarray = np.array(feature_list, dtype=np.float32)
    feature_ndarray = feature_ndarray.reshape((len(feature_list),))
    # ndarray类型转换为字符串
    feature_bytes = feature_ndarray.tobytes()
    feature_base64_bytes = base64.encodebytes(feature_bytes)
    feature_base64 = str(feature_base64_bytes.decode())
    return feature_base64


def as_num(x):
    y='{:.2f}'.format(x)
    return(y)

# 校验参数
def check_param(param):
    if param == "":
        return False
    else:
        return True

# 基础的返回
def return_base_result(code,message):
    result = {"code": code,
              "message": message
              }
    return Response(json.dumps(result), mimetype='application/json')

# 查询特征库状态返回
def return_get_feature_library_result(code,message,count):
    result = {"code": code,
              "message": message,
              "count": count}
    return Response(json.dumps(result), mimetype='application/json')


# 加载批量特征返回
def return_full_template_result(code,message,data):
    result = {"code": code,
              "message": message,
              "data": data}
    return Response(json.dumps(result), mimetype='application/json')

# 加载单条特征入库/删除特征入库返回
def return_feature_template_result(code,message,uuid,library):
    result = {"code": code,
              "message": message,
              "uuid": uuid,
              "library": library}
    return Response(json.dumps(result), mimetype='application/json')

# 查询特征返回
def return_search_feature_template_result(code,message,uuid,library,feature):
    result = {"code": code,
              "message": message,
              "uuid": uuid,
              "library": library,
              "feature": feature}
    return Response(json.dumps(result), mimetype='application/json')

# 1比N返回
def return_get_topN(code, message, total, ainumber, results):
    return_data = {"code": code,
              "message": message,
              "total": total,
              "ainumber": ainumber,
              "results": results}
    return Response(json.dumps(return_data), mimetype='application/json')


@app.route('/')
def hello_world():
    return 'Hello World!'


# 第一部分：库管理
# (1)建库
@app.route('/algorithm/loadFeatureLibrarys', methods=['POST'])
def load_feature_librarys():
    try:
        libraryNumber = str(request.json['libraryNumber'])
        if check_param(libraryNumber) is True:
            res = client.list_partitions(collection_name="face_vector")
            # 库存在
            if libraryNumber in res:
                code = 10002
                message = "{0}库已存在，创建失败".format(libraryNumber)
            # 库不存在
            else:
                # 创建分区：Create Partitions
                client.create_partition(
                    collection_name="face_vector",
                    partition_name=libraryNumber
                )
                code = 0
                message = "{0}库创建成功".format(libraryNumber)
        else:
            code = 10001
            message = "传入参数为空"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_base_result(code, message)

# 第一部分：库管理
# (2)查询特征库状态
@app.route('/algorithm/getFeatureLibrary', methods=['POST'])
def get_feature_library():
    try:
        libraryNumber = str(request.json['libraryNumber'])
        if check_param(libraryNumber) is True:
            res = client.list_partitions(collection_name="face_vector")
            # 库存在
            if libraryNumber in res:
                # 统计collection中分区的entities总数
                res = client.query(
                    collection_name="face_vector",
                    # highlight-start
                    output_fields=["count(*)"],
                    partition_names=[libraryNumber],
                    consistency_level=0
                    # highlight-end
                )
                code = 0
                message = "{0}库查询成功".format(libraryNumber)
                count = str(res[0]['count(*)'])
            else:
                code = 10003
                message = "{0}库不存在".format(libraryNumber)
                count = None
        else:
            code = 10001
            message = "传入参数为空"
            count = None
    except Exception as e:
        code = 20000
        message = str(e)
        count = None
    return return_get_feature_library_result(code, message, count)

# 第一部分：库管理
# (3)删除库
@app.route('/algorithm/deleteFeatureLibrary', methods=['POST'])
def delete_feature_library():
    try:
        libraryNumber = str(request.json['libraryNumber'])
        if check_param(libraryNumber) is True:
            res = client.list_partitions(collection_name="face_vector")
            # 库存在
            if libraryNumber in res:
                # 首先释放分区，然后再删除
                client.release_partitions(
                    collection_name="face_vector",
                    partition_names=[libraryNumber]
                )
                client.drop_partition(
                    collection_name="face_vector",
                    partition_name=libraryNumber
                )
                code = 0
                message = "{0}库删除成功".format(libraryNumber)
            else:
                code = 10003
                message = "{0}库不存在".format(libraryNumber)
        else:
            code = 10001
            message = "传入参数为空"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_base_result(code, message)

# 第二部分：模板管理
# (1)批量加载特征模板/修改特征
@app.route('/algorithm/loadFullTemplate', methods=['POST'])
def load_full_template():
    try:
        # 初始化返回的data_results字典
        data_results = {
            "result": {
                "success": {
                    "data": []
                },
                "failed": {
                    "data": []
                }
            }
        }
        # 接收的字段
        data = request.json['data']
        # 类型转换
        # 将data里面library、uuid、feature字段都强制转换为字符串类型
        data = [{k: str(v) for k, v in item.items()} for item in data]
        # 判断data里面所有的library值是否是一致
        library_group = data[0]["library"] if all(item["library"] == data[0]["library"] for item in data) else None
        # 将feature值都经过featureStringToVector函数进行处理
        for item in data:
            item["feature"] = featureStringToVector(item["feature"]).tolist()
        # 若加载的库都一样
        if library_group is not None:
            # 判断库/分区是否存在 True False
            has_partition_res = client.has_partition(
                collection_name="face_vector",
                partition_name=library_group
            )
            # 如果库/分区存在
            if has_partition_res == True:
                # 多条插入，自带更新数据
                insert_entity_res = client.insert(
                    collection_name="face_vector",
                    data=data,
                    partition_name=library_group
                )
                # res = {'insert_count': 10, 'cost': 0}
                success_total = insert_entity_res['insert_count']
                if success_total == len(data):
                    code = 0
                    message = "特征模板加载成功"
                    success_data = [{k: v for k, v in item.items() if k != "feature"} for item in data]
                    data_results["result"]["success"]["data"] = success_data
                else:
                    code = 10008
                    message = "特征模板加载失败"
                    failed_data = [{**{k: v for k, v in item.items() if k != "feature"}, "reason": str(code)} for item in data]
                    data_results["result"]["failed"]["data"] = failed_data

            # 如果库/分区不存在
            else:
                code = 10003
                message = "{0}库不存在".format(library_group)
                failed_data = [{**{k: v for k, v in item.items() if k != "feature"}, "reason": str(code)} for item in data]
                data_results["result"]["failed"]["data"] = failed_data
        else:
            code = 10009
            message = "加载库不一致"
            failed_data = [{**{k: v for k, v in item.items() if k != "feature"}, "reason": str(code)} for item in data]
            data_results["result"]["failed"]["data"] = failed_data
    except Exception as e:
        code = 20000
        message = str(e)
        data_results = None
    return return_full_template_result(code, message, data_results)


# (2)单条加载特征模板/修改特征
@app.route('/algorithm/loadFeatureTemplate', methods=['POST'])
def load_feature_template():
    try:
        insert_data = []
        entity = {}
        uuid = str(request.json['uuid'])
        library = str(request.json['library'])
        feature = str(request.json['feature'])
        embedding = featureStringToVector(feature).tolist()
        if check_param(library) is True and check_param(uuid) is True and check_param(feature) is True:
            entity['uuid'] = uuid
            entity['library'] = library
            entity['feature'] = embedding
            insert_data.append(entity)
            # 判断库/分区是否存在 True False
            has_partition_res = client.has_partition(
                collection_name="face_vector",
                partition_name=library
            )
            # 如果库/分区存在
            if has_partition_res == True:
                # 插入数据，单条插入，自带更新数据
                insert_entity_res = client.insert(
                    collection_name="face_vector",
                    data=insert_data,
                    partition_name=library
                )
                # res = {'insert_count': 10, 'cost': 0}
                success_total = insert_entity_res['insert_count']
                if success_total == 1:
                    code = 0
                    message = "特征模板加载成功"
                else:
                    code = 10008
                    message = "特征模板加载失败"
            # 如果库/分区不存在
            else:
                code = 10003
                message = "{0}库不存在".format(library)
        else:
            code = 10001
            message = "传入参数为空"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_feature_template_result(code, message, uuid, library)

# 第二部分：模板管理
# (3)删除特征模板
@app.route('/algorithm/deleteFeatureTemplate', methods=['POST'])
def delete_feature_template():
    try:
        uuid = str(request.json['uuid'])
        library = str(request.json['library'])
        if check_param(uuid) is True and check_param(library) is True:
            # 判断库/分区是否存在 True False
            has_partition_res = client.has_partition(
                collection_name="face_vector",
                partition_name=library
            )
            # 如果库/分区存在
            if has_partition_res == True:
                # 判断特征存不存在
                get_entity_res = client.get(
                    collection_name="face_vector",
                    partition_names=[library],
                    ids=[uuid]
                )
                # entity存在
                if len(get_entity_res) != 0:
                    # 删除特征模板
                    delete_entity_res = client.delete(
                        collection_name="face_vector",
                        partition_name=library,
                        ids=[uuid]
                    )
                    success_total = delete_entity_res['delete_count']
                    if success_total == 1:
                        code = 0
                        message = "特征模板删除成功"
                    else:
                        code = 10008
                        message = "特征模板删除失败"
                # entity不存在
                else:
                    code = 10007
                    message = "特征模板不存在"
            # 如果库/分区不存在
            else:
                code = 10003
                message = "{0}库不存在".format(library)
        else:
            code = 10001
            message = "传入参数为空"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_feature_template_result(code, message, uuid, library)

# 第二部分：模板管理
# (4)查询特征模板
@app.route('/algorithm/searchFeatureTemplate', methods=['POST'])
def search_feature_template():
    try:
        uuid = str(request.json['uuid'])
        library = str(request.json['library'])
        if check_param(uuid) is True and check_param(library) is True:
            # 判断库/分区是否存在 True False
            has_partition_res = client.has_partition(
                collection_name="face_vector",
                partition_name=library
            )
            # 如果库/分区存在
            if has_partition_res == True:
                # 判断特征存不存在
                get_entity_res = client.get(
                    collection_name="face_vector",
                    partition_names=[library],
                    ids=[uuid]
                )
                # entity存在
                if len(get_entity_res) != 0:
                    # 查询获取
                    get_feature_list = get_entity_res[0]['feature']
                    feature = featureListToString(get_feature_list)
                    code = 0
                    message = "特征模板查询成功"
                else:
                    code = 10007
                    message = "特征模板不存在"
                    feature = None
            # 如果库/分区不存在
            else:
                code = 10003
                message = "{0}库不存在".format(library)
                feature = None
        else:
            code = 10001
            message = "传入参数为空"
            feature = None
    except Exception as e:
        code = 20000
        message = str(e)
        feature = None
    return return_search_feature_template_result(code, message, uuid, library, feature)

# 第三部分：1比N搜索
# (1)图像搜索
@app.route('/algorithm/getTopNByImage', methods=['POST'])
def get_topN_by_image():
    try:
        results = []
        query_data = []
        query_image_base64 = str(request.json['query_image_base64'])
        repository_ids = str(request.json['repository_ids'])
        library = list(set(repository_ids.split(',')))
        top_N = int(request.json['top_n'])
        # 调用提取特征接口，提取特征值
        feature_extract_postJson = json.dumps({
            "imageBase64": query_image_base64
        }, cls=myJsonEncoder.MyEncoder)
        feature_extract_r = requests.post(feature_extract_url, feature_extract_postJson,
                                          headers={'Content-Type': 'application/json', 'Connection': 'close'})
        if feature_extract_r.status_code == requests.codes.ok:
            feature_extract_jsonResult = feature_extract_r.json()
            if (feature_extract_jsonResult["code"] == 0):
                # 提取到的特征
                query_feature_string = feature_extract_jsonResult["data"]["feature"]
                # 特征字符串类型转换为list类型
                query_feature_list = featureStringToVector(query_feature_string).tolist()
                query_data.append(query_feature_list)
                # 去库里面寻找相似的特征
                res = client.search(
                    collection_name="face_vector",
                    data=query_data,
                    limit=top_N,  # Max. number of search results to return
                    search_params={"metric_type": "L2", "params": {}},
                    partition_names=library,
                    output_fields=["library", "uuid"]
                )
                total = len(res[0])
                if total > 0:
                    code = 0
                    message = "图像搜索成功"
                    ainumber = "6"
                    for i in range(total):
                        result = {
                            'id_number': str(res[0][i]['entity']['uuid']),
                            'similarity': float(as_num(distcance_to_score(res[0][i]['distance']))),
                            'repository_id': str(res[0][i]['entity']['library'])
                        }
                        results.append(result)
                else:
                    code = 10005
                    message = "图像搜索失败"
                    total = None
                    ainumber = "6"
            else:
                code = 10004
                message = "提取特征失败"
                total = None
                ainumber = "6"
        else:
            code = 10006
            message = "提取特征服务异常"
            total = None
            ainumber = "6"
    except Exception as e:
        code = 20000
        message = str(e)
        total = None
        ainumber = "6"
    return return_get_topN(code, message, total, ainumber, results)


# 第三部分：1比N搜索
# (2)特征搜索
# 1比N特征检索
@app.route('/algorithm/getTopNByFeature', methods=['POST'])
def get_topN_by_feature():
    try:
        results = []
        query_data = []
        query_image_feature = str(request.json['query_image_feature'])
        repository_ids = str(request.json['repository_ids'])
        library = list(set(repository_ids.split(',')))
        top_N = int(request.json['top_n'])
        # 特征字符串类型转换为list类型
        query_feature_list = featureStringToVector(query_image_feature).tolist()
        query_data.append(query_feature_list)
        # 去库里面寻找相似的特征
        res = client.search(
            collection_name="face_vector",
            data=query_data,
            limit=top_N,  # Max. number of search results to return
            search_params={"metric_type": "L2", "params": {}},
            partition_names=library,
            output_fields=["library", "uuid"]
        )
        total = len(res[0])
        if total > 0:
            code = 0
            message = "特征搜索成功"
            ainumber = "6"
            for i in range(total):
                result = {
                    'id_number': str(res[0][i]['entity']['uuid']),
                    'similarity': float(as_num(distcance_to_score(res[0][i]['distance']))),
                    'repository_id': str(res[0][i]['entity']['library'])
                }
                results.append(result)
        else:
            code = 10005
            message = "特征搜索失败"
            total = None
            ainumber = "6"
    except Exception as e:
        code = 20000
        message = str(e)
        total = None
        ainumber = "6"
    return return_get_topN(code, message, total, ainumber, results)


if __name__ == '__main__':
    #app.run(host='192.168.9.194', port=7000, threaded=True, debug=False)
    app.run(host='0.0.0.0', port=6201)
    #app.run()
