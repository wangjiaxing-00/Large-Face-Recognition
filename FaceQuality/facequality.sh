cd /home/ai/project/Large-Face-Recognition/FaceQuality
/home/ai/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_0.py main:app & 
/home/ai/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_1.py main:app
