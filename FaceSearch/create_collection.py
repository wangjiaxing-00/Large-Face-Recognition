import os
from pymilvus import MilvusClient, DataType
import time

# 连接服务，初始化 MilvusClient
client = MilvusClient(
     uri="http://127.0.0.1:19530",
     db_name="face"
)

# 如果自增id的话，auto_id=True,否则为Flase
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
schema.add_field(field_name="uuid", datatype=DataType.VARCHAR, is_primary=True, max_length=255)
schema.add_field(field_name="library", datatype=DataType.VARCHAR, max_length=255)
schema.add_field(field_name="feature", datatype=DataType.FLOAT_VECTOR, dim=128)
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="uuid",
    index_name="uuid_index",
    index_type="INVERTED"
)
index_params.add_index(
    field_name="library",
    index_name="library_index",
    index_type="INVERTED"
)
index_params.add_index(
    field_name="feature",
    index_name="feature_index",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 4096, "nprobe": 128}
)
# 创建 collection
client.create_collection(
    collection_name="face_vector",
    schema=schema,
    index_params=index_params
)
time.sleep(5)

print("face_vector collection create complete...")



