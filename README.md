# Large-Face-Recognition

超大规模1比N人脸识别项目解决方案，支持数十亿级别库容规模，具备高并发，低延迟性能。

本项目包括FaceQuality（人脸图像质量评估）、FaceMatrix（人脸1比1）、taskScheduling（任务调度）、FaceSearch（人脸搜索）4个子项目。

- [x] FaceQuality：包括人脸图像质量总分、光照、清晰度、人脸姿态角度、是否佩戴口罩。

- [x] FaceMatrix：包括人脸提取特征值、人脸图像1比1、人脸特征1比1。

- [x] taskScheduling：包括任务分发、结果融合。

- [x] FaceSearch：包括人脸库管理（建库、删库、查询库）、特征模板管理（特征加载、特征删除、特征查询）、人脸1比N。

<div align="left">
  <img src="https://github.com/wangjiaxing-00/Large-Face-Recognition/blob/main/images/1.png" width="640"/>
</div>

# 开始

环境： 提前安装CUDA11.7+Cudnn8.9.2、milvus-gpu-2.4.5数据库（可视化可用Attu）

conda create -n insightface python=3.10

pip install -r requirements.txt

# 模型下载请联系作者，邮箱1197311349@qq.com

## 1、FaceQuality

## 模型

人脸检测模型：（1）face_detect_model/sd/sd_det.onnx  或者（2）face_detect_model/rf_landmark106

人脸质量评估模型：（1）EQFaceEQFace/backbone.pth、EQFace/quality.pth   或者  （2）SDD_FIQA/SDD_FIQA_checkpoints_r50.pth

人脸角度模型：FSANet/model/fsanet-1x1-iter-688590.onnx、FSANet/model/fsanet-var-iter-688590.onnx

口罩检测模型：FaceMask/last.onnx、FaceMask/coco.names

以上模型配置选择在文件config/algorithmPara.ini配置

## 运行

单卡运行：python main.py

多卡运行，修改facequality.sh内容、config/gunicorn_x.py，指定文件路径、运行环境、绑定显卡编号；修改main.py内容，取消注释最后一行app.run()，添加注释app.run(host='0.0.0.0', port=5001, threaded=False, debug=False)，最后执行bash facequality.sh

## 服务接口
url: http://ip:5000/faceQuality

POST请求：
{
    "query_image_base64":"图像base64"
}



