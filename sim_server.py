# sim_server.py
from flask import Flask
import time

app = Flask(__name__)

@app.route("/run",methods = ["GET"])
def run():
    # 用于测试服务是否并行
    time.sleep(1)
    return "0"

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=1472,debug=True)
    # gunicorn -w 4 -b 0.0.0.0:1472 sim_server:app
    # ab -c 4 -n 10 http://172.16.113.103:1472/run