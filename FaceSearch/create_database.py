from pymilvus import connections, db

conn = connections.connect(host="127.0.0.1", port=19530)
database = db.create_database("face")
print("face database create complete...")