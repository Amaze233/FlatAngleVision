'''
@Project ：FlatAngleWeb
@File    ：gunicorn.conf.py
@Author  ：谢逸帆
@Date    ：2021/7/15 10:23 
'''

bind = '0.0.0.0:5001'    # 绑定ip和监听端口
worker_class = "gevent"  # 异步模式
workers = 5             # 进程数 = cpu*2+1

backlog = 2048
max_requests = 1000
debug = True
daemon = True
reload = True
pidfile = 'log/gunicore.pid'
loglevel = 'log/debug.log'
accesslog = 'log/access.log'
errorlog = 'log/error.log'
