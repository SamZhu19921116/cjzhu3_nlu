#!/bin/sh
# gunicorn 关闭后不会自动删掉pid文件，这里自行删掉
project_path=/root/cjzhu3_nlu
flask_file=nlu_server_v202202281
cd ${project_path}

case $1 in
    start)
        gunicorn --logger-class 'gunicorn_conf.custom_logger' -c gunicorn_conf.py ${flask_file}:app # --preload
        echo ${project_path}/${flask_file}.py start!
        ;;
    stop)
        for id in `cat ${project_path}/log/gunicorn.pid`;do 
        kill -9 ${id} 
        done
        echo ${project_path}/${flask_file}.py stop!
        ;;
    restart)
        for id in `cat ${project_path}/log/gunicorn.pid`;do 
        kill -HUP ${id} 
        done
        echo ${project_path}/${flask_file}.py restart!
        ;;
    status)
        pstree -ap | grep gunicorn
        ;;
    log)
        tail -f ${project_path}/log/debug.log
        ;;
    *)
        echo "example:bash gunicorn_kill.sh [start|stop|restart|log]"
        ;;
esac
