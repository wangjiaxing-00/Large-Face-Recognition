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
import re
import myJsonEncoder
import requests, json
import concurrent.futures
import threading
from pymilvus import MilvusClient, DataType
from concurrent.futures import ThreadPoolExecutor

'''配置文件'''
config = configparser.ConfigParser()
config.read("config/algorithmPara.ini", "utf-8")
node_num = config.getint('node', 'node_num')
# 包含了http的基础地址
# base_http_dict = {'ip_1': 'http://192.168.9.194:8001',
#                   'ip_2': 'http://192.168.9.137:8001'}
base_http_dict = {}
for i in range(node_num):
    key = "ip_"+str((i+1))
    base_http_dict[key] = "http://"+config.get('ip', key)+":8001"
# 包含了库分配
# distribute_library_dict = {'distribute_library_1': ['11', '12', '13', '14', '15', 'test1'],
#                            'distribute_library_2': ['21', '22', '23', 'test2']}
distribute_library_dict = {}
for i in range(node_num):
    key = "distribute_library_"+str((i+1))
    distribute_library_dict[key] = config.get('distribute', key).split(',')

http_library_relationship_dict = {}
for i in range(node_num):
    key = "ip_"+str((i+1))
    http_library_relationship_dict[key] = "distribute_library_"+str((i+1))


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False  # False返回json不按照字母顺序


# 定义1比N调用接口的函数
def call_1vsN_api(url, payload, timeout):
    try:
        postJson = json.dumps(payload)
        r = requests.post(url, postJson, headers={'Content-Type': 'application/json'}, timeout=timeout)
        if r.status_code == requests.codes.ok:
            return r.json()
        else:
            return []
    except requests.exceptions.Timeout:
        return []

# 库分组处理-包括固定分配+最后一台机器分配新库的策略
def repository_ids_porcess(repository_ids):
    library = list(set(repository_ids.split(',')))
    # 创建一个新的字典来存储分组结果，键来自 distribute_library_list
    grouped_repository_ids = {key: [] for key in distribute_library_dict}
    # 遍历 library 中的每个元素，并根据 distribute_library_list 进行分组
    for id in library:
        for key, ids in distribute_library_dict.items():
            if id in ids:
                grouped_repository_ids[key].append(id)
                break  # 找到匹配的库后停止循环
    # 检查是否有未分组的元素，如果有，则可以选择添加到最后一个库或创建一个新库
    ungrouped_ids = [id for id in library if id not in sum(distribute_library_dict.values(), [])]
    if ungrouped_ids:
        # 这里选择添加到最后一个库，也可以选择创建一个新库
        last_key = list(distribute_library_dict.keys())[-1]
        grouped_repository_ids[last_key].extend(ungrouped_ids)
    # 将分组结果中的列表转换为逗号分隔的字符串
    for key in grouped_repository_ids:
        if grouped_repository_ids[key] != '':
            grouped_repository_ids[key] = ','.join(grouped_repository_ids[key])
    # 移除grouped_repository_ids中值为空的元素
    non_empty_library_dict = {key: value for key, value in grouped_repository_ids.items() if value.strip()}
    # 将 distribute_library_dict 中的键转换为 base_http_dict 中的键，并创建新的字典
    non_empty_base_http_dict = {key.replace('distribute_library_', 'ip_'):
                                    base_http_dict[key.replace('distribute_library_', 'ip_')]
                                for key in non_empty_library_dict}
    return non_empty_library_dict, non_empty_base_http_dict

# 融合函数
def merge_thread_results(thread_results):
    # 初始化一个空列表来存储所有有效的 results
    all_results = []
    has_success = False
    all_flag = 0  # 如果为0的话代表都正常
    # 遍历 thread_results 中的每个字典
    for result in thread_results:
        # 成功的
        if result['code'] == 0:
            has_success = True
            # 将有效的 results 添加到 all_results 列表中
            all_results.extend(result['results'])
            success_code = result['code']
            success_message = result['message']
            success_total = result['total']
            success_ainumber = result['ainumber']
        # 失败的
        else:
            all_flag = all_flag +1
            fail_code = result['code']
            fail_message = result['message']
            fail_total = result['total']
            fail_ainumber = result['ainumber']
            fail_selected_results = []
    if has_success:
        code = success_code
        message = success_message
        total = success_total
        ainumber = success_ainumber
        # 根据 similarity 对 all_results 进行排序，降序排列
        sorted_results = sorted(all_results, key=lambda x: x['similarity'], reverse=True)
        # 获取排序后前 total 个结果
        selected_results = sorted_results[:success_total]
    else:
        code = fail_code
        message = fail_message
        total = fail_total
        ainumber = fail_ainumber
        selected_results = fail_selected_results
    # 创建返回的字典结构
    merged_result = {
        'code': code,
        'message': message,
        'total': total,
        'ainumber': ainumber,
        'results': selected_results,
        'all_flag': all_flag
    }
    return merged_result


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
def return_get_topN(code, message, total, ainumber, results, all_flag):
    return_data = {"code": code,
              "message": message,
              "total": total,
              "ainumber": ainumber,
              "results": results,
              "all_flag": all_flag}
    return Response(json.dumps(return_data), mimetype='application/json')


@app.route('/')
def hello_world():
    return 'Hello World!'

# 第一部分：库管理
# (1)建库
@app.route('/algorithm/loadFeatureLibrarys', methods=['POST'])
def load_feature_librarys():
    try:
        # 初始化为False，表示未找到
        library_found = False
        # 入参
        libraryNumber = str(request.json['libraryNumber'])
        # 调用建库接口
        postJson = json.dumps({
            "libraryNumber": libraryNumber
        }, cls=myJsonEncoder.MyEncoder)
        for i in range(node_num):
            # 如果所创建的库在规划之中
            if libraryNumber in distribute_library_dict['distribute_library_'+str(i+1)]:
                library_found = True
                r = requests.post(base_http_dict['ip_'+str(i+1)]+'/algorithm/loadFeatureLibrarys',
                                                  postJson,
                                                  headers={'Content-Type': 'application/json', 'Connection': 'close'})
                if r.status_code == requests.codes.ok:
                    json_result = r.json()
                    code = json_result["code"]
                    message = json_result["message"]
                    break
                else:
                    code = 10006
                    message = base_http_dict['ip_'+str(i+1)]+"网络服务异常"
        # 如果所创建的库不在规划之中,则创建在最后一台服务器上
        if not library_found:
            r = requests.post(base_http_dict['ip_' + str(node_num)] + '/algorithm/loadFeatureLibrarys',
                              postJson,
                              headers={'Content-Type': 'application/json', 'Connection': 'close'})
            if r.status_code == requests.codes.ok:
                json_result = r.json()
                code = json_result["code"]
                message = json_result["message"]
            else:
                code = 10006
                message = base_http_dict['ip_'+str(node_num)]+"网络服务异常"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_base_result(code, message)

# 第一部分：库管理
# (2)查询特征库状态
@app.route('/algorithm/getFeatureLibrary', methods=['POST'])
def get_feature_library():
    try:
        # 初始化为False，表示未找到
        library_found = False
        libraryNumber = str(request.json['libraryNumber'])
        # 调用建库接口
        postJson = json.dumps({
            "libraryNumber": libraryNumber
        }, cls=myJsonEncoder.MyEncoder)
        for i in range(node_num):
            # 去规划库中寻找
            if libraryNumber in distribute_library_dict['distribute_library_'+str(i+1)]:
                library_found = True
                r = requests.post(base_http_dict['ip_'+str(i+1)]+'/algorithm/getFeatureLibrary',
                                                  postJson,
                                                  headers={'Content-Type': 'application/json', 'Connection': 'close'})
                if r.status_code == requests.codes.ok:
                    json_result = r.json()
                    code = json_result["code"]
                    message = json_result["message"]
                    count = json_result["count"]
                    break
                else:
                    code = 10006
                    message = base_http_dict['ip_'+str(i+1)]+"网络服务异常"
                    count = None
        # 去最后一台服务器寻找
        if not library_found:
            r = requests.post(base_http_dict['ip_' + str(node_num)] + '/algorithm/getFeatureLibrary',
                              postJson,
                              headers={'Content-Type': 'application/json', 'Connection': 'close'})
            if r.status_code == requests.codes.ok:
                json_result = r.json()
                code = json_result["code"]
                message = json_result["message"]
                count = json_result["count"]
            else:
                code = 10006
                message = base_http_dict['ip_' + str(node_num)] + "网络服务异常"
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
        # 初始化为False，表示未找到
        library_found = False
        libraryNumber = str(request.json['libraryNumber'])
        # 调用建库接口
        postJson = json.dumps({
            "libraryNumber": libraryNumber
        }, cls=myJsonEncoder.MyEncoder)
        for i in range(node_num):
            # 去规划库中删除
            if libraryNumber in distribute_library_dict['distribute_library_'+str(i+1)]:
                library_found = True
                r = requests.post(base_http_dict['ip_'+str(i+1)]+'/algorithm/deleteFeatureLibrary',
                                                  postJson,
                                                  headers={'Content-Type': 'application/json', 'Connection': 'close'})
                if r.status_code == requests.codes.ok:
                    json_result = r.json()
                    code = json_result["code"]
                    message = json_result["message"]
                    break
                else:
                    code = 10006
                    message = base_http_dict['ip_'+str(i+1)]+"网络服务异常"
        # 去最后一台服务器删除
        if not library_found:
            r = requests.post(base_http_dict['ip_' + str(node_num)] + '/algorithm/deleteFeatureLibrary',
                              postJson,
                              headers={'Content-Type': 'application/json', 'Connection': 'close'})
            if r.status_code == requests.codes.ok:
                json_result = r.json()
                code = json_result["code"]
                message = json_result["message"]
            else:
                code = 10006
                message = base_http_dict['ip_' + str(node_num)] + "网络服务异常"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_base_result(code, message)

# 第二部分：模板管理
# (1)批量加载特征模板/修改特征
@app.route('/algorithm/loadFullTemplate', methods=['POST'])
def load_full_template():
    try:
        # 初始化为False，表示未找到
        library_found = False
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
        # 将data里面library、uuid、feature字段都强制转换为字符串类型
        data = [{k: str(v) for k, v in item.items()} for item in data]
        # 判断data里面所有的library值是否是一致
        library_group = data[0]["library"] if all(item["library"] == data[0]["library"] for item in data) else None
        # 若加载的库都一样,那么调用服务
        if library_group is not None:
            # 调用建库接口
            postJson = json.dumps({
                "data": data
            }, cls=myJsonEncoder.MyEncoder)
            for i in range(node_num):
                # 去规划库中查询并加载特征
                if library_group in distribute_library_dict['distribute_library_' + str(i + 1)]:
                    library_found = True
                    r = requests.post(base_http_dict['ip_' + str(i + 1)] + '/algorithm/loadFullTemplate',
                                      postJson,
                                      headers={'Content-Type': 'application/json', 'Connection': 'close'})
                    if r.status_code == requests.codes.ok:
                        json_result = r.json()
                        code = json_result["code"]
                        message = json_result["message"]
                        data_results = json_result["data"]
                        break
                    else:
                        code = 10006
                        message = base_http_dict['ip_' + str(i + 1)] + "网络服务异常"
                        data_results = None
            # 去最后一台服务器查询并加载特征
            if not library_found:
                r = requests.post(base_http_dict['ip_' + str(node_num)] + '/algorithm/loadFullTemplate',
                                  postJson,
                                  headers={'Content-Type': 'application/json', 'Connection': 'close'})
                if r.status_code == requests.codes.ok:
                    json_result = r.json()
                    code = json_result["code"]
                    message = json_result["message"]
                    data_results = json_result["data"]
                else:
                    code = 10006
                    message = base_http_dict['ip_' + str(node_num)] + "网络服务异常"
                    data_results = None
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
        # 初始化为False，表示未找到
        library_found = False
        insert_data = []
        entity = {}
        uuid = str(request.json['uuid'])
        library = str(request.json['library'])
        feature = str(request.json['feature'])
        # 调用建库接口
        postJson = json.dumps({
            "uuid": uuid,
            "library": library,
            "feature": feature
        }, cls=myJsonEncoder.MyEncoder)
        for i in range(node_num):
            # 去规划库中查询并加载特征
            if library in distribute_library_dict['distribute_library_'+str(i+1)]:
                library_found = True
                r = requests.post(base_http_dict['ip_'+str(i+1)]+'/algorithm/loadFeatureTemplate',
                                                  postJson,
                                                  headers={'Content-Type': 'application/json', 'Connection': 'close'})
                if r.status_code == requests.codes.ok:
                    json_result = r.json()
                    code = json_result["code"]
                    message = json_result["message"]
                    uuid = json_result["uuid"]
                    library = json_result["library"]
                    break
                else:
                    code = 10006
                    message = base_http_dict['ip_'+str(i+1)]+"网络服务异常"
        # 去最后一台服务器查询并加载特征
        if not library_found:
            r = requests.post(base_http_dict['ip_' + str(node_num)] + '/algorithm/loadFeatureTemplate',
                              postJson,
                              headers={'Content-Type': 'application/json', 'Connection': 'close'})
            if r.status_code == requests.codes.ok:
                json_result = r.json()
                code = json_result["code"]
                message = json_result["message"]
            else:
                code = 10006
                message = base_http_dict['ip_' + str(node_num)] + "网络服务异常"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_feature_template_result(code, message, uuid, library)

# 第二部分：模板管理
# (3)删除特征模板
@app.route('/algorithm/deleteFeatureTemplate', methods=['POST'])
def delete_feature_template():
    try:
        # 初始化为False，表示未找到
        library_found = False
        uuid = str(request.json['uuid'])
        library = str(request.json['library'])
        # 调用建库接口
        postJson = json.dumps({
            "uuid": uuid,
            "library": library
        }, cls=myJsonEncoder.MyEncoder)
        for i in range(node_num):
            # 去规划库中查询并删除特征
            if library in distribute_library_dict['distribute_library_'+str(i+1)]:
                library_found = True
                r = requests.post(base_http_dict['ip_'+str(i+1)]+'/algorithm/deleteFeatureTemplate',
                                                  postJson,
                                                  headers={'Content-Type': 'application/json', 'Connection': 'close'})
                if r.status_code == requests.codes.ok:
                    json_result = r.json()
                    code = json_result["code"]
                    message = json_result["message"]
                    uuid = json_result["uuid"]
                    library = json_result["library"]
                    break
                else:
                    code = 10006
                    message = base_http_dict['ip_'+str(i+1)]+"网络服务异常"
        # 去最后一台服务器查询并加载特征
        if not library_found:
            r = requests.post(base_http_dict['ip_' + str(node_num)] + '/algorithm/deleteFeatureTemplate',
                              postJson,
                              headers={'Content-Type': 'application/json', 'Connection': 'close'})
            if r.status_code == requests.codes.ok:
                json_result = r.json()
                code = json_result["code"]
                message = json_result["message"]
            else:
                code = 10006
                message = base_http_dict['ip_' + str(node_num)] + "网络服务异常"
    except Exception as e:
        code = 20000
        message = str(e)
    return return_feature_template_result(code, message, uuid, library)

# 第二部分：模板管理
# (4)查询特征模板
@app.route('/algorithm/searchFeatureTemplate', methods=['POST'])
def search_feature_template():
    try:
        # 初始化为False，表示未找到
        library_found = False
        uuid = str(request.json['uuid'])
        library = str(request.json['library'])
        # 调用建库接口
        postJson = json.dumps({
            "uuid": uuid,
            "library": library
        }, cls=myJsonEncoder.MyEncoder)
        for i in range(node_num):
            # 去规划库中查询并查询特征
            if library in distribute_library_dict['distribute_library_'+str(i+1)]:
                library_found = True
                r = requests.post(base_http_dict['ip_'+str(i+1)]+'/algorithm/searchFeatureTemplate',
                                                  postJson,
                                                  headers={'Content-Type': 'application/json', 'Connection': 'close'})
                if r.status_code == requests.codes.ok:
                    json_result = r.json()
                    code = json_result["code"]
                    message = json_result["message"]
                    uuid = json_result["uuid"]
                    library = json_result["library"]
                    feature = json_result["feature"]
                    break
                else:
                    code = 10006
                    message = base_http_dict['ip_'+str(i+1)]+"网络服务异常"
                    feature = None
        # 去最后一台服务器查询并查询特征
        if not library_found:
            r = requests.post(base_http_dict['ip_' + str(node_num)] + '/algorithm/searchFeatureTemplate',
                              postJson,
                              headers={'Content-Type': 'application/json', 'Connection': 'close'})
            if r.status_code == requests.codes.ok:
                json_result = r.json()
                code = json_result["code"]
                message = json_result["message"]
                feature = None
            else:
                code = 10006
                message = base_http_dict['ip_' + str(node_num)] + "网络服务异常"
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
        # 接收入参
        query_image_base64 = str(request.json['query_image_base64'])
        repository_ids = str(request.json['repository_ids'])
        top_N = int(request.json['top_n'])
        # 库编号分组处理-根据指定规则分组到不同机器上
        non_empty_library_dict, non_empty_base_http_dict = repository_ids_porcess(repository_ids)
        # 创建body体
        payloads = {key: {'query_image_base64': query_image_base64,
                          'repository_ids': non_empty_library_dict[http_library_relationship_dict[key]],
                          'top_n': top_N}
                    for key in non_empty_base_http_dict.keys()}
        # 准备并行调用
        with ThreadPoolExecutor() as executor:
            futures = {}
            for server, url in non_empty_base_http_dict.items():
                payload = payloads.get(server, {})  # 获取对应服务器的 payload
                futures[executor.submit(call_1vsN_api, url+'/algorithm/getTopNByImage', payload, 3)] = server
            # thread_results存储收集结果
            thread_results = []
            for future in concurrent.futures.as_completed(futures):
                thread_results.append(future.result())
        merged_result = merge_thread_results(thread_results)
        code = merged_result['code']
        message = merged_result['message']
        total = merged_result['total']
        ainumber = merged_result['ainumber']
        results = merged_result['results']
        all_flag = merged_result['all_flag']
    except Exception as e:
        code = 20000
        message = str(e)
        total = None
        ainumber = "6"
        results = []
        all_flag = -1
    return return_get_topN(code, message, total, ainumber, results, all_flag)

# 第三部分：1比N搜索
# (2)特征搜索
# 1比N特征检索
@app.route('/algorithm/getTopNByFeature', methods=['POST'])
def get_topN_by_feature():
    try:
        query_image_feature = str(request.json['query_image_feature'])
        repository_ids = str(request.json['repository_ids'])
        top_N = int(request.json['top_n'])
        # 库编号分组处理-根据指定规则分组到不同机器上
        non_empty_library_dict, non_empty_base_http_dict = repository_ids_porcess(repository_ids)
        # 创建body体
        payloads = {key: {'query_image_feature': query_image_feature,
                          'repository_ids': non_empty_library_dict[http_library_relationship_dict[key]],
                          'top_n': top_N}
                    for key in non_empty_base_http_dict.keys()}
        # 准备并行调用
        with ThreadPoolExecutor() as executor:
            futures = {}
            for server, url in non_empty_base_http_dict.items():
                payload = payloads.get(server, {})  # 获取对应服务器的 payload
                futures[executor.submit(call_1vsN_api, url + '/algorithm/getTopNByFeature', payload, 3)] = server
            # thread_results存储收集结果
            thread_results = []
            for future in concurrent.futures.as_completed(futures):
                thread_results.append(future.result())
        merged_result = merge_thread_results(thread_results)
        code = merged_result['code']
        message = merged_result['message']
        total = merged_result['total']
        ainumber = merged_result['ainumber']
        results = merged_result['results']
        all_flag = merged_result['all_flag']
    except Exception as e:
        code = 20000
        message = str(e)
        total = None
        ainumber = "6"
        results = []
        all_flag = -1
    return return_get_topN(code, message, total, ainumber, results, all_flag)

if __name__ == '__main__':
    #app.run(host='192.168.9.194', port=7000, threaded=True, debug=False)
    app.run(host='0.0.0.0', port=8000)
    #app.run()


