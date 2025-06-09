创建数据库：运行create_database.py    face数据库
创建集合：运行create_collection.py    face_vector集合
# 第一部分：库管理
[1] 建库
http://127.0.0.1:700x/algorithm/loadFeatureLibrarys
入参：
{
    "libraryNumber":"xx"
}
返回：
{
    "code":0,
    "message":"xx库创建成功"
}

[2] 查询库状态
http://127.0.0.1:700x/algorithm/getFeatureLibrary
入参：
{
    "libraryNumber":"xx"
}
返回：
{
    "code":0,
    "message":"xx库查询成功",
    "count": "xxxx"
}

[3] 删除库
http://127.0.0.1:700x/algorithm/deleteFeatureLibrary
入参：
{
    "libraryNumber":"xx"
}
返回：
{
    "code":0,
    "message":"xx库删除成功"
}

# 第二部分：模板管理
[1] 批量加载特征/修改特征
http://127.0.0.1:700x/algorithm/loadFullTemplate
入参：
{
    "data":[
          {
            "library":xx,
            "uuid":"",
            "feature":""
          },
          {
            "library":xx,
            "uuid":"",
            "feature":""
          }
    ]
}
返回：
{
    "code":0,
    "message":"特征模板加载成功",
    "data":{
        "result":{
            "success":{
                "data":[
                    {
                        "uuid":"",
                        "library":xx
                    },
                    {
                        "uuid":"",
                        "library":xx
                    }
                ]
            },
            "failed":{
                "data":[
                        {
                            "uuid":"",
                            "library":xx,
                            "reason":""
                        },
                        {
                            "uuid":"",
                            "library":xx,
                            "reason":""
                        }
                    ]
            }
        }
    }
}

[2] 单条加载特征/修改特征
http://127.0.0.1:700x/algorithm/loadFeatureTemplate
入参：
{
    "library":xx,
    "uuid":"",
    "feature":""
}
返回：
{
    "code":0,
    "message":"特征模板加载成功",
    "uuid":"",
    "library":xx
}



[2] 删除特征
http://127.0.0.1:700x/algorithm/deleteFeatureTemplate
入参：
{
    "uuid":"xx",
    "library":"xx"
}
返回：
{
    "code":0,
    "message":"特征模板删除成功",
    "uuid": "xx",
    "library": "xx"
}

[3] 查询特征
http://127.0.0.1:700x/algorithm/searchFeatureTemplate
入参：
{
    "uuid":"xx",
    "library":"xx"
}
返回：
{
    "code":0,
    "message":"特征模板查询成功",
    "uuid":"xx",
    "library":"xx",
    "feature":"xx"
}

# 第三部分：1比N搜索
[1] 图像搜索
http://127.0.0.1:700x/algorithm/getTopNByImage
入参：
{
    "query_image_base64":"xx",
    "repository_ids":"xx,xx,xx", # test
    "top_n":"10"
}
返回：
{
    "code":0,
    "message":"图像搜索成功",
    "total":xx,
    "ainumber":"6",
    "results":[
        {
          "id_number":"xx",
          "similarity":98.52,
          "repository_id":"xx"
        }
        {
          "id_number":"xx",
          "similarity":92.52,
          "repository_id":"xx"
        }
    ]
}

[2] 特征搜索
http://127.0.0.1:700x/algorithm/getTopNByFeature
入参：
{
    "query_image_feature""xx",
    "repository_ids":"xx,xx,xx",
    "top_n":"10"
}
返回：
{
    "code":0,
    "message":"特征搜索成功",
    "total": xx,
    "ainumber": "6",
    "results": [
        {
          "id_number": "xx",
          "similarity": 98.52,
          "repository_id": "xx"
        }
        {
          "id_number": "xx",
          "similarity": 92.52,
          "repository_id": "xx"
        }
    ]
}


修改索引：运行modify_index.py

任务调度项目taskScheduling