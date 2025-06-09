# Large-Face-Recognition

超大规模1比N人脸识别项目解决方案，支持数十亿级别库容规模，具备高并发，低延迟性能。

本项目包括FaceQuality（人脸图像质量评估）、FaceMatrix（人脸1比1）、FaceTask（人脸调度）、FaceSearch（人脸搜索）4个子项目。More actions

- [x] FaceQuality：包括人脸图像质量总分、光照、清晰度、人脸姿态角度、是否佩戴口罩。

- [x] FaceMatrix：包括人脸提取特征值、人脸图像1比1、人脸特征1比1。

- [x] FaceTask：包括人脸任务分发、人脸结果融合。

- [x] FaceSearch：包括人脸库管理（建库、删库、查询库）、特征模板管理（特征加载、特征删除、特征查询）、人脸1比N。

<div align="left">
  <img src="https://github.com/wangjiaxing-00/Large-Face-Recognition/blob/main/images/1.png" width="640"/>
</div>

# 开始

环境： 提前安装python3.10、CUDA11.7+Cudnn8.9.2、milvus-gpu-2.4.5数据库（可视化可用Attu）

conda create -n python=3.10
pip install -r requirements
