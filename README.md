# Large-Face-Recognition

超大规模1比N人脸识别项目解决方案，支持数十亿级别库容规模，具备高并发，低延迟性能。

本项目包括FaceQuality（人脸图像质量评估）、FaceMatrix（人脸1比1）、taskScheduling（任务调度）、FaceSearch（人脸搜索）4个子项目。

- [x] FaceQuality：包括人脸图像质量总分、光照、清晰度、人脸姿态角度、是否佩戴口罩。

- [x] FaceMatrix：包括人脸提取特征值、人脸图像1比1、人脸特征1比1。

- [x] taskScheduling：包括任务分发、结果融合。

- [x] FaceSearch：包括人脸库管理（建库、删库、查询库）、特征模板管理（特征加载、特征删除、特征查询）、人脸1比N。

<div align="left">
  <img src="https://github.com/wangjiaxing-00/Large-Face-Recognition/blob/main/images/1.png" width="640"/>
  <img src="https://github.com/wangjiaxing-00/Large-Face-Recognition/blob/main/images/2.png" width="640"/>
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

以上部分模型配置选择在文件config/algorithmPara.ini配置

## 运行

方式一：修改main.py内容，取消注释app.run(host='0.0.0.0', port=5001, threaded=False, debug=False)，注释app.run()，最后执行python main.py

方式二：（1）修改facequality.sh内容、config/gunicorn_x.py，指定文件路径、运行环境、绑定显卡编。（2）修改main.py内容，取消注释app.run()，注释app.run(host='0.0.0.0', port=5001, threaded=False, debug=False)，最后执行bash facequality.sh，适合多机多卡

## 服务接口
url: http://ip:端口/faceQuality

POST请求：
{
    "query_image_base64":"图像base64"
}

## 2、FaceMatrix

## 模型

人脸检测模型：（1）face_detect_model/sd/sd_det.onnx  或者（2）face_detect_model/rf_landmark106

人脸识别模型：face_recognition_model/500w_r100_128D/500w_r100_128D.onnx

以上部分模型配置选择在文件config/algorithmPara.ini配置

## 运行

方式一：修改main.py内容，取消注释app.run(host='0.0.0.0', port=6001, threaded=False, debug=False)，注释app.run()，最后执行python main.py

方式二：（1）修改1vs1.sh内容、config/gunicorn_x.py，指定文件路径、运行环境、绑定显卡编。（2）修改main.py内容，取消注释app.run()，注释app.run(host='0.0.0.0', port=6001, threaded=False, debug=False)，最后执行bash 1vs1.sh，适合多机多卡

## 服务接口

图片比图片url: http://ip:端口/algorithm/compareImageAndImage

图片比特征url: http://ip:端口/algorithm/compareImageAndFeature

图片生成特征值url: http://ip:端口/algorithm/getFeature

批量图片生成特征值url: http://ip:端口/algorithm/getBatchFeature

特征值比特征值url: http://ip:端口/algorithm/compareFeatureAndFeature

详细post请求格式见代码main.py和app_run.py内容
