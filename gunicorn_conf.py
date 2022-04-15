# gunicorn.conf
import os
import gevent.monkey
gevent.monkey.patch_all()

import multiprocessing

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT_PATH, 'log')

if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

# 120.26.167.21:机器配置
# 查看物理CPU的个数:cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l :1个物理CPU
# 查看CPU是几核:cat /proc/cpuinfo | grep "cores" | uniq :6核心
# 查看逻辑CPU的个数:cat /proc/cpuinfo | grep "processor" | wc -l :12线程数

debug = False
bind = "0.0.0.0:1472" # 绑定ip和端口号
timeout = 100 # 超时
reload = True
daemon = True # True意味着开启后台运行，默认为False
threads = 20 # 指定每个进程开启的线程数
workers = 10 # 启动的进程数 multiprocessing.cpu_count() * 2 + 1 # 推荐核数*2+1发挥最佳性能
worker_class = 'gunicorn.workers.ggevent.GeventWorker' # 使用gevent模式，还可以使用sync 模式，默认的是sync模式
worker_connections = 300
x_forwarded_for_header = 'X-FORWARDED-FOR'

loglevel = 'INFO' # 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
pidfile = os.path.join(LOG_PATH, "gunicorn.pid") # 存放Gunicorn进程pid的位置，便于跟踪
accesslog = os.path.join(LOG_PATH, "access.log")  # 访问日志文件
errorlog = os.path.join(LOG_PATH, "debug.log") # 错误日志文件

# 自定义gunicorn日志输入格式
import logging
from gunicorn import glogging
class custom_logger(glogging.Logger):
    # Custom logger for Gunicorn log messages
    def setup(self, cfg):
        # Configure Gunicorn application logging configuration
        super().setup(cfg)
        # Override Gunicorn's `error_log` configuration.
        self._set_handler(self.error_log, cfg.errorlog, logging.Formatter(fmt=('timestamp:%(asctime)s||pid:%(process)d||loglevel:%(levelname)s||code:%(filename)s-%(funcName)s-%(lineno)s||msg:%(message)s')))
