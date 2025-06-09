from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from pymilvus import connections, db

# connections.connect("default", host="localhost", port="19530")
# # 列出当前所有collection
# print(utility.list_indexes(collection_name="test_collection"))
# print(utility.index_building_progress(collection_name="test_collection",index_name="vector"))

connections.connect(host="127.0.0.1", port="19530")
db.using_database("face")
# 列出当前所有collection
print(utility.list_indexes(collection_name="face_vector"))
print(utility.index_building_progress(collection_name="face_vector",index_name="feature_index"))
