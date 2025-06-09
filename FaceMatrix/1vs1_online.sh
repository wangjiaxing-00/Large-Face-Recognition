cd /home/program/self/code/FaceMatrix
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_0.py main:app & 
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_1.py main:app &
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_2.py main:app &
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_3.py main:app &
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_4.py main:app &
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_5.py main:app &
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_6.py main:app &
/home/program/self/anaconda3/envs/insightface/bin/gunicorn -c config/gunicorn_7.py main:app
