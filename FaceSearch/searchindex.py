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


