import os
from pymilvus import MilvusClient, DataType
import time

# 连接服务，初始化 MilvusClient
client = MilvusClient(
     uri="http://127.0.0.1:19530",
     db_name="face"
)

# 释放集合
client.release_collection(
     collection_name="face_vector"
 )

#删除集合
client.drop_collection(
     collection_name="face_vector"
)