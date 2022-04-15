# -*- coding: utf-8 -*-
import json
import logging
import os
import re
from signal import SIGHUP
from flask import Flask, request

from MODEL_ENGINE_V2 import Model_Engine
from RULE_ENGINE import Rule_Engine
from SLOT_ENGINE import Slot_Engine

app=Flask(__name__) # 创建新的开始
app.config['JSON_AS_ASCII'] = False # 解决中文乱码问题

from RULENER_ENGINE_V1 import register_type,RuleNER

register_type("surname",r"(赵|钱|孙|李|周|吴|郑|王|冯|陈|褚|卫|蒋|沈|韩|杨|朱|秦|尤|许|何|吕|施|张|孔|曹|严|华|金|魏|陶|姜|戚|谢|邹|喻|柏|水|窦|章|云|苏|潘|葛|奚|范|彭|郎|鲁|韦|昌|马|苗|凤|花|方|俞|任|袁|柳|酆|鲍|史|唐|费|廉|岑|薛|雷|贺|倪|汤|滕|殷|罗|毕|郝|邬|安|常|乐|于|时|傅|皮|卞|齐|康|伍|余|元|卜|顾|孟|平|黄|和|穆|萧|尹|姚|邵|湛|汪|祁|毛|禹|狄|米|贝|明|臧|计|伏|成|戴|谈|宋|茅|庞|熊|纪|舒|屈|项|祝|董|梁|杜|阮|蓝|闵|席|季|麻|强|贾|路|娄|危|江|童|颜|郭|梅|盛|林|刁|钟|徐|邱|骆|高|夏|蔡|田|樊|胡|凌|霍|虞|万|支|柯|昝|管|卢|莫|经|房|裘|缪|干|解|应|宗|丁|宣|贲|邓|郁|单|杭|洪|包|诸|左|石|崔|吉|钮|龚|程|嵇|邢|滑|裴|陆|荣|翁|荀|羊|於|惠|甄|曲|家|封|芮|羿|储|靳|汲|邴|糜|松|井|段|富|巫|乌|焦|巴|弓|牧|隗|山|谷|车|侯|宓|蓬|全|郗|班|仰|秋|仲|伊|宫|宁|仇|栾|暴|甘|钭|厉|戎|祖|武|符|刘|景|詹|束|龙|叶|幸|司|韶|郜|黎|蓟|薄|印|宿|白|怀|蒲|台|从|鄂|索|咸|籍|赖|卓|蔺|屠|蒙|池|乔|阴|欎|胥|能|苍|双|闻|莘|党|翟|谭|贡|劳|逄|姬|申|扶|堵|冉|宰|郦|雍|郤|璩|桑|桂|濮|牛|寿|通|边|扈|燕|冀|郏|浦|尚|农|温|别|庄|晏|柴|瞿|阎|充|慕|连|茹|习|宦|艾|鱼|容|向|古|易|慎|戈|廖|庾|终|暨|居|衡|步|都|耿|满|弘|匡|国|文|寇|广|禄|阙|东|殴|殳|沃|利|蔚|越|夔|隆|师|巩|厍|聂|晁|勾|敖|融|冷|訾|辛|阚|那|简|饶|空|曾|毋|沙|乜|养|鞠|须|丰|巢|关|蒯|相|查|后|荆|红|游|竺|权|逯|盖|益|桓|公|万俟|司马|上官|欧阳|夏侯|诸葛|闻人|东方|赫连|皇甫|尉迟|公羊|澹台|公冶|宗政|濮阳|淳于|单于|太叔|申屠|公孙|仲孙|轩辕|令狐|钟离|宇文|长孙|慕容|鲜于|闾丘|司徒|司空|亓官|司寇|仉|督|子车|颛孙|端木|巫马|公西|漆雕|乐正|壤驷|公良|拓跋|夹谷|宰父|谷梁|晋|楚|闫|法|汝|鄢|涂|钦|段干|百里|东郭|南门|呼延|归海|羊舌|微生|岳|帅|缑|亢|况|郈|有琴|梁丘|左丘|东门|西门|商|牟|佘|佴|伯|赏|南宫|墨|哈|谯|笪|年|爱|阳|佟|第五|言|福|百|姓)")

register_type("do_surname",r"(姓|叫|免贵姓|信)")

rn = RuleNER()
rn.add_rule("extract_surname","*{do_surname:do_surname}{surname:surname}*")

# 槽值请求字典处理
model_slot =  os.path.join(os.path.dirname(__file__),"slot_module","model_slot")
if not os.path.exists(model_slot):
    os.makedirs(model_slot)

rule_slot =  os.path.join(os.path.dirname(__file__),"slot_module","rule_slot")
if not os.path.exists(rule_slot):
    os.makedirs(rule_slot)

slot_engine = Slot_Engine(model_dir = model_slot,rule_dir = rule_slot)

############################Keyword_Semantic模块###############################
# 规则资源及模型资源加载
import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file=r'/data/jdduan/data/cls/vocab.txt', do_lower_case=True)
hnsw_model = os.path.join(os.path.dirname(__file__),"hnsw_model")
model_engine = Model_Engine(tokenizer, hnsw_model)
rule_model = os.path.join(os.path.dirname(__file__),"rules_model")
rule_engine = Rule_Engine(rule_model)

def Auto_Sort_Keyword_Semantic_V12(parent_node, query, model_score = 0.7, rule_score = 0.7):
    # query = Str_Clean(query)
    # {"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}
    # {}
    FineSortRule = rule_engine.rules_parse_v1(parent_node, query, filter_score = rule_score)
    # app.logger.info("规则结果:{}".format(FineSortRule))
    # {"match_score":FineSortAvg['avg_score'],"match_node":FineSortAvg['intent'],"match_rule":FineSortAvg['retrieved'],"match_module":"avg_score"}
    # {}
    FineSortModel = model_engine.models_parse_v2(parent_node, query, filter_score = model_score)
    # app.logger.info("模型结果:{}".format(FineSortList))
    FineSortList = []
    if FineSortModel and float(FineSortModel['match_score']) >= 0.99: # 资源库原句评估
        predict_intent = {"module":FineSortModel["match_module"], "rule":FineSortModel["match_rule"], "kp":FineSortModel["match_node"], "score":str(FineSortModel["match_score"])}
        FineSortList.append(FineSortModel)
        if FineSortRule:
            FineSortList.append(FineSortRule)
    else: # 未匹配到原句则加入规则结果融合排序
        if FineSortModel: # 模型不为空
            FineSortList.append(FineSortModel)
        if FineSortRule: # 规则不为空
            FineSortList.append(FineSortRule)
        if len(FineSortList) != 0: # 模型 + 规则不为空
            FineSortRes = sorted(FineSortList, key=lambda x:x['match_score'], reverse = True)[0]
            predict_intent = {"module":FineSortRes["match_module"], "rule":FineSortRes["match_rule"], "kp":FineSortRes["match_node"], "score":str(FineSortRes["match_score"])}
        else:
            predict_intent = {}

    # 对query长度小于等于2的特殊后处理
    if len(query) <= 2 and predict_intent and float(predict_intent['score']) < 0.99:
        predict_intent = {}

    co_exist_intents = [{"kp":item['match_node'], "module":item['match_module'], "rule":item['match_rule'], "score":str(item['match_score'])} for item in FineSortList]
    return predict_intent, co_exist_intents

@app.route('/test')
def test():
    return '<h1>Hello World</h1>'

@app.route('/hnsw/update',methods=['GET','POST'])
def hnsw_update():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        app.logger.info("语料更新请求:{}".format(req))
        json_data = json.loads(req)
        org_name = json_data['org'] # 父亲节点
        slot_map = json_data['slot_map']
        # 将语料中申请的槽值提取字典保存至指定文件中:slot_model/org_name.pkl(此项功能仅为适配RDG接口，暂时仅开放语料中槽值提取功能)
        # 暂时仅提供规则提槽工具
        slot_engine.model_slot_update(slot_map,model_slot,org_name)
        res = json_data['kps'] # 父亲节点对应所有子节点资源内容
        model_engine.hnsw_index_update(res, hnsw_model, org_name)
        # # 方案一:通过向该进程的父进程下的所有子进程发送kill，然后master进程重新启动新的子进程
        # import psutil
        # cur_pid = psutil.Process()
        # par_pid = psutil.Process(cur_pid.ppid())
        # son_pids = [i.pid for i in par_pid.children()]
        # app.logger.info("pid:{},ppid:{},sonpid".format(cur_pid.pid,cur_pid.ppid(),son_pids))
        # for pid in son_pids:
        #     if pid != cur_pid.pid:
        #         os.kill(pid,SIGKILL)
        #         app.logger.info("kill pid:{}".format(pid))
        # 方案二: 向master进程发送HUP信号：重新加载配置，使用新配置启动新的工作进程并优雅地关闭旧的工作进程。如果应用程序没有预加载（使用preload_app选项），Gunicorn 也会加载它的新版本。
        os.kill(os.getppid(),SIGHUP)
        return {"status": 0, "result": res}

@app.route('/hnsw/delete',methods=['GET','POST'])
def hnsw_delete():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        parent_node = json_data['org']
        model_engine.hnsw_index_delete(org_dir = hnsw_model,org_name = parent_node)
        # 方案二: 向master进程发送HUP信号：重新加载配置，使用新配置启动新的工作进程并优雅地关闭旧的工作进程。如果应用程序没有预加载（使用preload_app选项），Gunicorn 也会加载它的新版本。
        # 禁止与--preload参数组合使用
        os.kill(os.getppid(),SIGHUP)

    return {'status':0}

# curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org":"开场白","query": "骗人的吧"}' http://127.0.0.1:1472/hnsw/search
@app.route('/hnsw/search',methods=['GET','POST'])
def hnsw_search():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        org_name = json_data['org'] # 父节点名称："开场白_yB7b2E"
        query = json_data['query'] # 用户query:骗人的吧
        res = model_engine.search(org_name,query,2)
        res = res.to_json(force_ascii=False) #解决DataFrame中文to_json乱码 
        return res

# 更新规则接口
@app.route('/rule/update',methods=['GET','POST'])
def rule_update():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        app.logger.info("规则更新请求:{}".format(req))
        json_data = json.loads(req)
        org_name,res,slot_map = parse_rdg_interface(json_data)
        # 将规则中申请的槽值提取字典保存至指定文件中:slot_model/org_name.pkl(此项功能仅为适配RDG接口，暂时不开放)
        slot_engine.rule_slot_update(slot_map,rule_slot,org_name) # 该接口目前有问题
        for parent_node,resource in res.items():
            rule_engine.rules_update(resource,rule_model,parent_node)
            # app.logger.info("输入规则节点:{},输入规则文本:{}".format(org_name,resource))
        # # 方案一:通过向该进程的父进程下的所有子进程发送kill，然后master进程重新启动新的子进程#
        # import psutil
        # cur_pid = psutil.Process()
        # par_pid = psutil.Process(cur_pid.ppid())
        # son_pids = [i.pid for i in par_pid.children()]
        # app.logger.info("pid:{},ppid:{},sonpid".format(cur_pid.pid,cur_pid.ppid(),son_pids))
        # for pid in son_pids:
        #     if pid != cur_pid.pid:
        #         os.kill(pid,SIGKILL)
        #         app.logger.info("kill pid:{}".format(pid)) 
        # 方案二: 向master进程发送HUP信号：重新加载配置，使用新配置启动新的工作进程并优雅地关闭旧的工作进程。如果应用程序没有预加载（使用preload_app选项），Gunicorn 也会加载它的新版本。
        # 禁止与--preload参数组合使用
        os.kill(os.getppid(),SIGHUP)
        return {"result": [], "msg": "success", "status": 0}

@app.route('/rule/delete',methods=['GET','POST'])
def rule_delete():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        parent_node = json_data['org']
        rule_engine.rules_delete(org_dir = rule_model,org_name = parent_node)
        # 方案二: 向master进程发送HUP信号：重新加载配置，使用新配置启动新的工作进程并优雅地关闭旧的工作进程。如果应用程序没有预加载（使用preload_app选项），Gunicorn 也会加载它的新版本。
        # 禁止与--preload参数组合使用
        os.kill(os.getppid(),SIGHUP)

    return {'status':0}

# 规则检查接口
@app.route('/rule/check',methods=['GET','POST'])
def rule_check():
        if request.method == 'POST':
            req = request.get_data(as_text=True)
            json_data = json.loads(req)
            status,rules = rule_engine.rules_check_v1(json_data)
            app.logger.info("规则检查请求:{},规则检查结果:{}".format(rules,status))
            return {"result":{"rules":rules,"status":json.loads(json.dumps(status))}}

# 规则查询接口
@app.route('/rule/search',methods=['GET','POST'])
def rule_search():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        query = json_data['query']
        parent_node = json_data['org']
        res = rule_engine.parse(parent_node,query)
        return res.to_json(force_ascii=False) # 解决DataFrame中文to_json乱码

# 解析RDG请求body
def parse_rdg_interface(json_data):
        parent_dict = {}
        org_name = None
        for key,value in json_data['org'].items():
            org_name = key # 父节点名
            parent_node_resources = value['resources']
            son_node_rule = parent_node_resources['rule'] # 子节点资源
            slot_map = parent_node_resources['slot_map']
            son_dict = {}
            for son_node_json in son_node_rule:
                son_node_name = son_node_json['kp']
                son_node_rules = son_node_json['rule']
                rule_arr = []
                for rule in son_node_rules:
                    if rule is not None:
                        tmp_rule = rule.split("/")
                        rule_arr.append({'rule':tmp_rule[0],'weight':tmp_rule[1]})
                son_dict[son_node_name] = rule_arr
            parent_dict[org_name] = son_dict
        return org_name, parent_dict, slot_map

@app.route('/nlu/search',methods=['GET','POST'])
def search():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        org_name = json_data['org'][0]['org_name'] # 父节点名称："开场白_yB7b2E"
        cls_score = json_data['org'][0]['cls_score'] # 模型类过滤阈值
        rule_score = json_data['org'][0]['rule_score'] # 规则类过滤阈值
        query = str(json_data['query']) # 用户query:骗人的吧
        sem_res, co_sem_res, ner_res= None,None,{}
        try:
            sem_res,co_sem_res = Auto_Sort_Keyword_Semantic_V12(parent_node = org_name, query = query, model_score = cls_score, rule_score = rule_score)
        except Exception as e:
            app.logger.exception(e)
            result = {"module":"none","org":org_name,"predict_intent":{},"query":query,"recommend":[],"slot":ner_res,"emotion":{},"co_exist_intents":[]}
        
        # 暂时仅开放规则接口
        slot_dict = slot_engine.rule_slot_module_dict
        if sem_res:
            try:
                for item in slot_dict[org_name]:
                    slot_name = item['slot_name']
                    # if_use_model = item['if_use_model']
                    if_use_rule = item['if_use_rule']
                    if if_use_rule and item['kp'] == sem_res['kp']:
                        for ent in rn.extract_entities(query, as_json=False):
                            ner_res[slot_name] = ent.entity_value
            except Exception as e:
                app.logger.exception(e)

        # 暂时不开放ner模型
        # try:
        #     for item in slot_dict[org_name]:
        #         kp = item['kp']
        #         # slot_name = item['slot_name']
        #         if_use_model = item['if_use_model']
        #         # if_use_rule = item['if_use_rule']
        #         if if_use_model and kp == sem_res['kp']:
        #             # ner_res = Ner_Predict(query,model,tokenizer)
        #             sem_res["slot"]={}
        # except Exception as e:
        #     app.logger.exception(e)

        if sem_res: #如果解析结果字典不为空
            result = {"module":sem_res['module'],"org":org_name,"predict_intent":sem_res,"query":query,"recommend":[],"slot":ner_res,"emotion":{},"co_exist_intents":co_sem_res}
        else: #如果解析结果字典为空
            result = {"module":"none","org":org_name,"predict_intent":{},"query":query,"recommend":[],"slot":ner_res,"emotion":{},"co_exist_intents":[]}

        app.logger.info("语义请求:{},语义响应:{}".format(data,result))
        return {"result":result,"status":0}

if __name__ != "__main__":
    # 如果不是直接运行，则将日志输出到 gunicorn 中
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

if __name__ == '__main__':
    # 通过本地端口映射启动并连接远程jupyter服务器
    # 1、在本地机器上进行 ssh 端口映射
    # 在CMD命令行中输入:ssh -p 22 -L 8008:127.0.0.1:8888 root@120.26.167.21
    # -p 21 表示远程访问的端口，有的可能不是21，需要换成端口22表示ssh而不是ftp
    # -L 8008:127.0.0.1:8888表示将远程服务器的 Jupyter Lab 端口8888 映射到本地机器 127.0.0.1 的 8008 端口
    # jupyter notebook --ip 0.0.0.0 --allow-root 
    # 2、在本地浏览器中输入127.0.0.1:8008打开页面

    # sem_res,co_sem_res = Auto_Sort_Keyword_Semantic_V11("开场白","我不要谢谢")
    # print(sem_res)
    # linux环境下查看机器性能：
    # 查看物理CPU的个数:cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l
    # 查看逻辑CPU的个数:cat /proc/cpuinfo | grep "processor" | wc -l
    # 查看CPU是几核:cat /proc/cpuinfo | grep "cores" | uniq
    # 查看CPU的主频:cat /proc/cpuinfo | grep MHz | uniq

    # cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l ==> 1个物理CPU
    # cat /proc/cpuinfo | grep "core id" | sort | uniq | wc -l ==> 6核数(每个物理CPU中core的个数(即核数))
    # cat /proc/cpuinfo | grep "processor" | sort | uniq | wc -l ==> 12个逻辑CPU

    # kernprof -l -v nlu_server_v202202081.py >nlu_server_v202202081.log
    handler = logging.FileHandler('flask.log', encoding='UTF-8') # 设置日志字符集和存储路径名字 
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s')) # 设置日志格式
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0',port=1472,debug=True) # 运行开始 #阿里云机器需要申请1472端口防火墙开启
    # app.run(host='0.0.0.0',port=8888,debug=True) # 运行开始 #阿里云机器需要申请1472端口防火墙开启
    # app.run(host='localhost',port=1472,debug=True,processes=True) # 运行开始 #阿里云机器需要申请1472端口防火墙开启
    # 内网地址：http://172.16.113.103:1472/
    # 公网地址：http://120.26.167.21:1472/

    # ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
    # curl -X POST http://120.26.167.21:1472/nlu/search -d '{"query": "你不要诈骗了","org": [{"org_name": "开场白","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}'
    # curl -X POST http://120.26.167.21:1472/nlu/search -d '{"query": "你不要诈骗了","org": [{"org_name": "开场白_yB7b2E","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}'
    # curl -X POST http://localhost:1472/nlu/search -d '{"query": "不需要","org": [{"org_name": "开场白","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}'
    # 加--preload 可以查到代码具体错误
    # gunicorn --logger-class 'gunicorn_conf.custom_logger' -c gunicorn_conf.py nlu_server_v202202281:app --preload
    # gunicorn -c gunicorn_conf.py nlu_server_v202202281:app --preload
    # ps ax | grep gunicorn
    # pstree -ap | grep gunicorn : 获取 Gunicorn 进程树
    # kill -9 24810 : 彻底杀死 Gunicorn 服务
    # kill -HUP 24810 : 重启 Gunicorn 服务

    # ab -n200 -c4 -T application/json -p post.json  http://localhost:1472/nlu/search
    # 报错：apr_socket_recv: Connection refused (111)
    # 修改ip地址解决问题：ab -n500 -c10 -T application/json -p post.json  http://127.0.0.1:1472/nlu/search 

    # 本地网压测
    # ab -n800 -c200 -T application/json -p post.json  http://127.0.0.1:1472/nlu/search
    # ab -n100000 -c100 -T application/json -p post.json  http://127.0.0.1:1472/nlu/search #Time per request:193.396 [ms] (mean)
    # torchserve压测
    # ab -k -l -n200 -c100 -p torchserve.json -T application/json http://127.0.0.1:8080/explanations/sbert
    
    # 公网压测引擎整体性能
    # nohup ./wrk -t10 -c100 -d 30m --latency --timeout 5s -s post.lua http://120.26.167.21:1472/nlu/search >t10_c100_d30m_all202102231355.txt &
    # nohup ./wrk -t10 -c150 -d 30m --latency --timeout 5s -s post.lua http://120.26.167.21:1472/nlu/search >t10_c150_d30m_all202102231630.txt &
    # nohup ./wrk -t10 -c200 -d 30m --latency --timeout 5s -s post.lua http://120.26.167.21:1472/nlu/search >t10_c200_d30m_all202102231435.txt &

    # 内网压测引擎整体性能
    # nohup ./wrk -t10 -c100 -d 30m --latency --timeout 5s -s post.lua http://127.0.0.1:1472/nlu/search >t10_c100_d30m_all202102231550.txt &
    # nohup ./wrk -t10 -c150 -d 30m --latency --timeout 5s -s post.lua http://127.0.0.1:1472/nlu/search >t10_c150_d30m_all202102231630.txt &
    # nohup ./wrk -t10 -c200 -d 30m --latency --timeout 5s -s post.lua http://127.0.0.1:1472/nlu/search >t10_c200_d30m_all202102231705.txt &

    # nohup ./wrk -t10 -c100 -d 30m --latency --timeout 5s -s kaichangbai15.lua http://127.0.0.1:1472/nlu/search >kaichangbai15_t10_c100_d30m_all202203091650.txt &

    # nohup ./wrk -t10 -c200 -d 30m --latency --timeout 5s -s kaichangbai15.lua http://127.0.0.1:1472/nlu/search >kaichangbai15_t10_c200_d30m_all202203100945.txt &

    # nohup ./wrk -t10 -c100 -d 30m --latency --timeout 5s -s ramdom_org.lua http://127.0.0.1:1472/nlu/search >ramdom_org_t10_c100_d30m_all202203091755.txt &

    # nohup ./wrk -t10 -c200 -d 30m --latency --timeout 5s -s ramdom_org.lua http://127.0.0.1:1472/nlu/search >ramdom_org_t10_c200_d30m_all202203101025.txt &

    # ./wrk -t1 -c1 -d 1s --latency --timeout 5s -s kaichangbai_org_corpus.lua http://127.0.0.1:1472/hnsw/update

    # ab -k -l -n1 -c1 -p kaichangbai_org_corpus.json -T application/json http://127.0.0.1:1472/hnsw/update

    # 内网压测bert模型接口整体性能
    # nohup ./wrk -t10 -c200 -d 30m --latency --timeout 5s -s tfserve32.lua http://127.0.0.1:8501/v1/models/model_cls_slot:predict >t10_c100_d60m_bert202102241048.txt &

    # rdg模型压测 max_seq_len：32, 字符长度：17-2 = 15
    # ab -k -l -n10000 -c200 -p rdg15.json -T application/json http://127.0.0.1:8501/v1/models/model_cls_slot:predict

    # max_seq_len：32, 字符长度：4-2 = 2
    # curl -X POST http://127.0.0.1:8501/v1/models/model_cls_slot:predict -d '{"instances": [{"ori_input_quests": [101,1998,3739,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}'

    # rdg模型压测 max_seq_len：32, 字符长度：4-2 = 2
    # ab -k -l -n10000 -c200 -p rdg2.json -T application/json http://127.0.0.1:8501/v1/models/model_cls_slot:predict

    # max_seq_len：32, 字符长度：17-2 = 15
    # curl -X POST http://127.0.0.1:8501/v1/models/model_cls_slot:predict -d '{"instances":[{"ori_input_quests":[101,2772,5442,3300,11567,7309,7579,100,6432,1139,3341,100,2600,5543,6237,1104,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}'

    # curl -X POST http://172.31.185.201:8501/v1/models/model_cls_slot:predict -d '{"instances":[{"ori_input_quests":[101,2772,5442,3300,11567,7309,7579,100,6432,1139,3341,100,2600,5543,6237,1104,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}'

    # aitalk外呼机器人测试环境：https://aitalk-test.5sale.cn/#/m/lexeme/list/l
    # 账号及密码：admin tszmlogin001