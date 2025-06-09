import os
os.environ['CUDA_VISIBLE_DEVICES']='7'
import multiprocessing
import gevent.monkey
gevent.monkey.patch_all()
import multiprocess

debug = True # 每次启动会显示，配置参数，生产环境可以设置为Ｆalse
#preload_app =True # 预加载资源
# 绑定ip+端口
bind = "0.0.0.0:6108"
# 进程数　＝　CPU数量×2+1
workers = 3
# 线程数 =  CPU数量×2
threads = multiprocessing.cpu_count()*2

# 等待队列最大长度，超过这个长度的链接将被拒绝连接
backlog = 2048
# 工作模式--协程
worker_class = 'gevent'
# 最大客户客户端并发数量，对使用线程和协程的worker的工作影响
# 服务器配置设置的值 1200：中小型项目 上万并发：中大型
# 服务器硬件：宽带+数据库+内存
# 服务器的架构：集群 主从
worker_connections = 1200
timeout = 500
keepalive = 100
# 进程名称
#proc_name = 'gunicorn_6002.pid'
# 进程pid记录文件
#pidfile = 'logs/app_run.logs'
# 日志等级
loglevel = 'error'
# 日志文件名
logfile = 'logs/syserror.logs'
# 访问记录
#accesslog = 'logs/access.logs'
# 访问记录格式
#acccess_log_format = '%(h)s %(t)s %(U)s %(q)s'
