# -*- coding: utf-8 -*-
import json
import logging
import os
import re
from crypt import methods

import numpy as np
import torch
from flask import Flask, request
from transformers import BertTokenizer

import args
# from RECALL_ENGINE import Recall_Engine
from BERT_NER.models.bert_for_ner import BertSoftmaxForNer
from BERT_NER.processors.ner_seq import ner_processors
from MODEL_ENGINE import Model_Engine
from RULE_ENGINE import Rule_Engine
from SLOT_ENGINE import Slot_Engine

app=Flask(__name__) # 创建新的开始
app.config['JSON_AS_ASCII'] = False # 解决中文乱码问题

##################################BERT_NER模块#######################################
sner_model = os.path.join(os.path.dirname(__file__),"BERT_NER/outputs/sner_output/bert")
tokenizer = BertTokenizer.from_pretrained(sner_model, do_lower_case="do_lower_case")
model = BertSoftmaxForNer.from_pretrained(sner_model)
model.to("cpu")
processor = ner_processors["sner"]()
label_list = processor.get_labels()
id2label = {i: label for i, label in enumerate(label_list)}

# Ner对文本预处理
def Ner_Preprocess(tokenizer, sentence:str):
    """ preprocess """
    tokens= tokenizer.tokenize(sentence)
    # insert "[CLS]"
    tokens.insert(0,"[CLS]")
    # insert "[SEP]"
    tokens.append("[SEP]")
    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < 128:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    input_ids_tensor = torch.tensor([input_ids],dtype=torch.long)
    input_mask_tensor = torch.tensor([input_mask],dtype=torch.long)
    segment_ids_tensor = torch.tensor([segment_ids],dtype=torch.long)
    return input_ids_tensor, input_mask_tensor, segment_ids_tensor

# Ner预测槽值及位置
def Ner_Predict(_sentence, _model = model, _tokenizer = tokenizer):
    input_ids,attention_mask,token_type_ids = Ner_Preprocess(_tokenizer,_sentence)
    input = {'input_ids':input_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids}
    res = []
    with torch.no_grad():
        outputs = _model(**input)
        logits = outputs[0]
        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=2).tolist()
        preds = preds[0][1:-1] # [CLS]XXXX[SEP]
        tags = [id2label[x] for x in preds]
        for index,char in enumerate(list(_sentence)):
            if tags[index] != 'O':
                res.append({"value":char, "slot_name":tags[index], "start_pos":index, "end_pos":index+1})
        return res

# 槽值请求字典处理
slot_model =  os.path.join(os.path.dirname(__file__),"corpus_slot_model")
slot_engine = Slot_Engine(slot_model)

import jieba

if os.path.exists("/tmp/jieba.cache"):
    os.remove("/tmp/jieba.cache") #每次删除结巴缓存
jieba.load_userdict("/usr/local/anaconda3/envs/nlu/lib/python3.7/site-packages/jieba/dict.txt")
from gensim import models

w2v_model = models.KeyedVectors.load(args.sort_word2vec_model_zhongan)

############################Keyword_Semantic模块###############################
# 加载双塔BERT预训练模型
import tokenization

tokenizer = tokenization.FullTokenizer(vocab_file=r'/data/jdduan/data/cls/vocab.txt', do_lower_case=True)
hnsw_model = os.path.join(os.path.dirname(__file__),"hnsw_model")
# hnsw_engine = Recall_Engine(tokenizer, w2v_model, jieba, hnsw_model)
model_engine = Model_Engine(tokenizer, w2v_model, jieba, hnsw_model)
rule_model = os.path.join(os.path.dirname(__file__),"rules_model")
rule_engine = Rule_Engine(rule_model)

# 精排过滤高得分召回数据
def FineSort_Filter(dict_data, filter_threshold = 0.99):
    if float(dict_data['match_score']) >= filter_threshold:
        return dict_data

# 通过正则算法过滤文本中的非法字符
def Str_Clean(sentence):
    sentence = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+","", sentence)
    return sentence

def Auto_Sort_Keyword_Semantic_V11(parent_node, query, model_score = 0.8, rule_score = 0.7):
    # query = Str_Clean(query)
    FineSortRes = None
    # {"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}
    # {}
    FineSortRule = rule_engine.rules_parse_v1(parent_node, query, filter_score = rule_score)
    # app.logger.info("规则结果:{}".format(FineSortRule))
    # {"match_score":FineSortSbert['sbert_score'],"match_node":FineSortSbert['intent'],"match_rule":FineSortSbert['retrieved'],"match_module":"sbert"}
    # {"match_score":FineSortW2V['w2v_score'],"match_node":FineSortW2V['intent'],"match_rule":FineSortW2V['retrieved'],"match_module":"word2vec"}
    # {"match_score":FineSortDiff['diff_score'],"match_node":FineSortDiff['intent'],"match_rule":FineSortDiff['retrieved'],"match_module":"difflib"}
    FineSortList = model_engine.models_parse_v1(parent_node, query, filter_score = model_score)
    # app.logger.info("模型结果:{}".format(FineSortList))

    # 资源库原句评估
    FineSortTmp = list(filter(lambda x: FineSort_Filter(x, 0.99), FineSortList))
    if len(FineSortTmp) != 0: # 匹配到原句则直接取原句作为结果
        FineSortRes = sorted(FineSortTmp, key=lambda x:x['match_score'], reverse = True)[0]
        if FineSortRule: # 如果规则结果不为空
            FineSortList.append(FineSortRule)
    else: # 未匹配到原句则加入规则结果融合排序
        if FineSortRule: # 如果规则结果不为空
            FineSortList.append(FineSortRule)
        if len(FineSortList) != 0:
            FineSortRes = sorted(FineSortList, key=lambda x:x['match_score'], reverse = True)[0]

    if FineSortRes: # 如果最终结果字典不为空
        predict_intent = {"module":FineSortRes["match_module"], "rule":FineSortRes["match_rule"], "kp":FineSortRes["match_node"], "score":str(FineSortRes["match_score"])}
    else:
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
        slot_engine.slot_update(slot_map, slot_model, org_name)
        res = json_data['kps'] # 父亲节点对应所有子节点资源内容
        model_engine.hnsw_index_update(res, hnsw_model, org_name)
        return {"status": 0, "result": res}

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
        res,slot_map = parse_rdg_interface(json_data)
        # 将规则中申请的槽值提取字典保存至指定文件中:slot_model/org_name.pkl(此项功能仅为适配RDG接口，暂时不开放)
        # slot_engine.slot_update(slot_map, slot_model, org_name)
        for org_name,resource in res.items():
            rule_engine.rules_update(resource,rule_model,org_name)
            # app.logger.info("输入规则节点:{},输入规则文本:{}".format(org_name,resource))
        return {"result": [], "msg": "success", "status": 0}

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

# 解析研究院请求body
def parse_rdg_interface(json_data):
        parent_dict = {}
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
        return parent_dict, slot_map

@app.route('/nlu/search',methods=['GET','POST'])
def search():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        org_name = json_data['org'][0]['org_name'] # 父节点名称："开场白_yB7b2E"
        cls_score = json_data['org'][0]['cls_score'] # 模型类过滤阈值
        rule_score = json_data['org'][0]['rule_score'] # 规则类过滤阈值
        query = json_data['query'] # 用户query:骗人的吧
        sem_res,co_sem_res = None,None
        try:
            sem_res,co_sem_res = Auto_Sort_Keyword_Semantic_V11(parent_node = org_name, query = query, model_score = cls_score, rule_score = rule_score)
        except Exception as e:
            app.logger.exception(e)
        
        slot_dict = slot_engine.slot_module_dict
        ner_res = {}
        try:
            for item in slot_dict[org_name]:
                kp = item['kp']
                slot_name = item['slot_name']
                if_use_model = item['if_use_model']
                if_use_rule = item['if_use_rule']
                if if_use_model and kp == sem_res['kp']:
                    ner_res = Ner_Predict(query,model,tokenizer)
                    sem_res["slot"]=ner_res
        except Exception as e:
            app.logger.exception(e)

        if sem_res: #如果解析结果字典不为空
            result = {"module":sem_res['module'],"org":org_name,"predict_intent":sem_res,"query":query,"recommend":[],"slot":ner_res,"emotion":{},"co_exist_intents":co_sem_res}
        else: #如果解析结果字典为空
            result = {"module":"none","org":org_name,"predict_intent":sem_res,"query":query,"recommend":[],"slot":ner_res,"emotion":{},"co_exist_intents":co_sem_res}

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
    # gunicorn --logger-class 'gunicorn_conf.custom_logger' -c gunicorn_conf.py nlu_server_v202202171:app --preload
    # gunicorn -c gunicorn_conf.py nlu_server_v202202171:app --preload
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

    # 内网压测bert模型接口整体性能
    # nohup ./wrk -t10 -c200 -d 30m --latency --timeout 5s -s tfserve32.lua http://127.0.0.1:8501/v1/models/model_cls_slot:predict >t10_c100_d60m_bert202102241048.txt &

    # rdg模型压测
    # ab -k -l -n10000 -c200 -p rdg.json -T application/json http://127.0.0.1:8501/v1/models/model_cls_slot:predict
    # curl -X POST http://127.0.0.1:8501/v1/models/model_cls_slot:predict -d '{"instances": [{"ori_input_quests": [101,1998,3739,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}'

    # curl -X POST http://127.0.0.1:8501/v1/models/model_cls_slot:predict -d '{"instances": [{"ori_input_quests": [101,1998,3739,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}]}'

    # aitalk外呼机器人测试环境：https://aitalk-test.5sale.cn/#/m/lexeme/list/l
    # 账号及密码：admin tszmlogin001
