import os
import base64
import numpy as np

# 特征字符串转换为向量
def featureStringToVector(feature):
    feature_base64_bytes = feature.encode()
    feature_bytes = base64.decodebytes(feature_base64_bytes)
    feature_numpy = np.frombuffer(feature_bytes, dtype=np.float32)
    return feature_numpy
def distance(embeddings1,embeddings2,distance_metric=0,alpha=-4.83702,beta=5.56252):
    if distance_metric == 0:
        diff = np.subtract(embeddings1,embeddings2)
        dist = np.sum(np.square(diff),0)
        dist = 1-(1./(1+np.exp(alpha*dist+beta)))
    elif distance_metric == 1:
        dot = np.sum(np.multiply(embeddings1,embeddings2),axis=0)
        norm = np.linalg.norm(embeddings1,axis=0)* np.linalg.norm(embeddings2,axis=0)
        similarity = dot/norm
        dist = similarity
    return dist

# 特征列表转换为特征字符串
def featureListToString(feature_list):
    # list类型转换为ndarray类型
    feature_ndarray = np.array(feature_list, dtype=np.float32)
    feature_ndarray = feature_ndarray.reshape((len(feature_list),))
    # ndarray类型转换为字符串
    feature_bytes = feature_ndarray.tobytes()
    feature_base64_bytes = base64.encodebytes(feature_bytes)
    feature_base64 = str(feature_base64_bytes.decode())
    return feature_base64

base64_feature_1 = "Mb+uO/n7d7x7jqk8D5D5Pa3WgL3KbiM9TtF4PMIpQrs6UOW4WNAqPJbvXTqK2Qq+6J/5PeN5az3R\nS0W90LqmPZ2/vb1JDYA9/f4dvVNdyzzSbIy9EFy+vBVYu72KLdc9N9oJvixSGD3Urbg7lbIEPvoD\nCr53dqm9nEtbPs7FwT3wvYw9k0H1vQ7Mkb1PENO90pXqvUv8Az2FfKW8ryjFPRPKjLtZDmq9F9Y3\nPed1Fj79Sys84zntParjjz3AYiq+GBQiupVziTw9rSc9KwAtvcCgo7sw57u9s5GFveqF77wgRgy9\nsgcuvptHt72vexa95KrBPYnWjj2rrZO9xByKPcJSs70yFUi9w4g5vWw0XT6V3vs9Gt/JvT3ajrsb\nxrg9wNJJvI7lM77pV5I9yCCsvLRuy7pYLjG7JruHPFqZ+Twf6mi8KRmQvMRCXL0cExe+DL1qPVWY\nRj2ixB69mJUQvWjakD2wdrQ9teIlPXN3Gr4HHly9w8fiO7Ytpz1F/jS9I6E3vX5zWj07X469k9qA\nPe7+jz47TuG9wdh4vSUQKj0mWgg8ZqPZPWeNUT1NxYe98dQWPj+Swr1rB1o+1D39PNUbpT3ctac9\neCYqPv0bcb0BdFy9pV34vY2upj0CliO+5WGlPWlS4r0rj0+7nehGvf36jb3HN927oPUdvg0Pij0=\n"
# base64特征字符串-->向量
feature_1 = featureStringToVector(base64_feature_1)
print("feature_1:", feature_1)  # {ndarray:(128,)}
print("type(feature_1):", type(feature_1))  # type(feature_1): <class 'numpy.ndarray'>
print("feature_1.dtype：", feature_1.dtype)
# ndarray--->>list--->>darray
feature_1_list = feature_1.tolist()
print("feature_1_list:", feature_1_list)
print("type(feature_1_list):", type(feature_1_list))  #  <class 'list'>

base64_feature_2 = featureListToString(feature_1_list)
print("base64_feature_2:", base64_feature_2)

# base64特征字符串-->向量
feature_2 = featureStringToVector(base64_feature_2)
print("feature_2:", feature_2)  # {ndarray:(256,)}
print("type(feature_2):", type(feature_2))  # type(feature_2): <class 'numpy.ndarray'>

sim = distance(feature_1, feature_2, 0)
print(sim)  #ValueError: operands could not be broadcast together with shapes (128,) (256,)

repository_ids ="11,12,13"
library = list(set(repository_ids.split(',')))
print(list(library))
