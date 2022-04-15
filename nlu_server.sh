#!/bin/bash

APP_NAME=nlu_server_v202201181.py
source activate
conda activate nlu

case $1 in
    start)
        nohup python ${APP_NAME} & #>/dev/null 2>&1 &
        echo ${APP_NAME} start!
        ;;
    stop)
        ps -ef | grep ${APP_NAME}| grep 1472 | grep -v grep | awk '{print $2}' | sed -e "s/^/kill -9 /g" | sh -
	ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
        echo ${APP_NAME} stop!
        ;;
    restart)
        bash "$0" stop
        sleep 3
        bash "$0" start
        ;;
    status)
        pstree -ap | grep gunicorn
        ;;
    log)
        tail -f nohup.out
        ;;
    *)
        echo "example:bash nlu_server.sh [start|stop|restart|status|log]" ;;
esac
