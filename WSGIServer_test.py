# coding: utf-8
# flask + gevent + multiprocess + wsgi

from gevent import monkey
from gevent.pywsgi import WSGIServer
monkey.patch_all()

import datetime
import os
from multiprocessing import cpu_count, Process
from flask import Flask, jsonify


app = Flask(__name__)

@app.route("/cppla", methods=['GET'])
def function_benchmark():
    print("function_benchmark")
    return jsonify(
        {
            "status": "ok",
            "time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            "pid": os.getpid()
        }
    ), 200

def run(MULTI_PROCESS):
    if MULTI_PROCESS == False:
        WSGIServer(('0.0.0.0', 1472), app).serve_forever()
    else:
        mulserver = WSGIServer(('0.0.0.0', 1472), app)
        mulserver.start()

        def server_forever():
            mulserver.start_accepting()
            mulserver._stop_event.wait()

        for i in range(cpu_count()):
            p = Process(target=server_forever)
            p.start()

if __name__ == "__main__":
    # 单进程 + 协程
    #run(False)
    # 多进程 + 协程
    run(True)
