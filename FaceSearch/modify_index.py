from pymilvus import MilvusClient, DataType, db
import time


# 连接服务，初始化 MilvusClient
client = MilvusClient(
     uri="http://127.0.0.1:19530",
     db_name="face"
)

# 如何修改索引
# Euclidean distance (L2)
# Inner product (IP)
# Cosine similarity (COSINE)

# FLAT
# IVF_FLAT：nlist，聚类时总的分桶数；nprobe：查询时需要搜索的分桶数目
# IVF_SQ8
# IVF_PQ
# GPU_IVF_FLAT
# GPU_IVF_PQ
# HNSW
# DISKANN

# 1b=10亿， 1m=100万
# 2,000,000条128维向量的数据大小约为1G，单条向量128*4=512B≈0.5KB
# 参数一：nlist:4096      nprobe:128
# 参数二：nlist:8192      nprobe:256
# 参数三：nlist:16384     nprobe:512

# 查看详细索引
res = client.describe_index(
    collection_name="face_vector",
    index_name="feature_index"
)
print("第一步，查看feature_index详细索引:", res)


# 释放集合
client.release_collection(
     collection_name="face_vector"
 )
res = client.get_load_state(
     collection_name="face_vector"
 )
print("第二步，释放集合，查看释放集合状态:", res)

# 先删除索引
client.drop_index(
     collection_name="face_vector",
     index_name="feature_index"
 )
print("第三步，删除索引完成")

# 要为集合创建索引或为集合建立索引，我们需要设置索引参数并调用create_index()。
# 设置索引的参数
index_params = MilvusClient.prepare_index_params()
# 在向量字段 vector 上面添加一个索引
index_params.add_index(
    field_name="feature",
    metric_type="L2",
    index_type="IVF_FLAT",
    index_name="feature_index",
    params={"nlist": 16384, "nprobe": 2048}
)
# 在集合创建索引文件
client.create_index(
    collection_name="face_vector",
    index_params=index_params
)
# 查看详细索引
res = client.describe_index(
    collection_name="face_vector",
    index_name="feature_index"
)
print("第四步，新索引创建完成，查看feature_index新索引:", res)

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
    print("第五步，加载集合")

