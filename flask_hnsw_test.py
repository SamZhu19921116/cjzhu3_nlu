from flask import Flask, request, jsonify
import pandas as pd
from hnsw_test import HNSW_SBERT_SERVER
from Robot_Keyword_Credit import rule_engine
import json
import os
import args
# 加载双塔BERT预训练模型
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer(args.retrival_sbert)
# 基于多种度量方式的计算两句子的相似度
from Sort_Similarity_Calculater import Similarity_Measure
sim_calc = Similarity_Measure()
# 双塔BERT文本相似度度量
from Sentence_Bert_Calculater import SBertSimCalc
sbert_sim = SBertSimCalc(sbert_model=sbert_model)

hnsw_model = os.path.join(os.path.dirname(__file__),"hnsw_model")
index = HNSW_SBERT_SERVER(sbert_model,hnsw_model)
rule_model = os.path.join(os.path.dirname(__file__),"rules_model","rule.bin")
rengine = rule_engine(rule_model)

# 召回topN问题及精排序(按节点分层，每层分不同召回集)
def Fine_Sort_V2(parent_node_name,query,topN = 10):
    #query问题粗召回
    RoughRecall = pd.DataFrame.from_records({'query': [query]*topN ,'retrieved': index.search(parent_node_name, query, topN)['custom'] , 'intent': index.search(parent_node_name, query, topN)['intent']}) 
    # 多种度量方式计算query问题以及query召回问题的相似度
    FineSort = pd.DataFrame.from_records(RoughRecall.apply(lambda row: sim_calc.Similarity_Calculate_ZhongAn_V2(row['query'], row['retrieved'], row['intent']), axis=1))
    FineSort['sbert_score'] = FineSort.apply(lambda row: sbert_sim.predict(row['query'], row['retrieved']) , axis=1)
    return FineSort

def Auto_Sort_Keyword_Semantic_V21(parent_node,query):
    # {"score":score_max,"hit_node":matcher_son_node,"match_query":matcher_son_rule}
    RuleParse = rengine.rule_parse(parent_node,query)
    RuleSort = RuleParse.head(1)
    # 规则类的匹配DataFrame:[score,hit_node,match_query]
    df_list = []
    df_list.append(RuleSort)
    # 粗召回+精排序DataFrame:[score,hit_node,match_query]
    FineSort = Fine_Sort_V2(parent_node_name=parent_node,query=query)
    # SBERT
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortSbert.iloc[0,-1],"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},index = [1]))
    # WORD2VEC
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortW2V.iloc[0,-1],"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]},index = [2]))
    res_df = pd.concat(df_list, ignore_index=True)
    result = res_df.sort_values(by=['score'],ascending=False).head(1)
    finesort_intent_score = result.iloc[0,-3] # 精排最终得分
    finesort_intent_node = result.iloc[0,-2] # 精排命中节点
    finesort_intent_match_query = result.iloc[0,-1] # 精排命中关键词集

    res_list = [
        {"rule_score":str(RuleParse.iloc[0,-3]),"hit_node":RuleSort.iloc[0,-2],"match_query":RuleParse.iloc[0,-1]},
        {"sbert_score":str(FineSortSbert.iloc[0,-1]),"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},
        {"w2v_score":str(FineSortW2V.iloc[0,-1]),"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]}
    ]
    return {"match_node":finesort_intent_node,"match_query":finesort_intent_match_query,"match_score":str(finesort_intent_score),"detail":res_list}


app=Flask(__name__) #创建新的开始

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"model_name": "BertClsModel","org": "开场白_yB7b2E","slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true}],"kps": {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"],"没小孩-幼儿3-8岁_EAdQu1": ["我都没有对象，怎么会有小孩","我儿子都二十多岁了","我们家宝宝三岁了","我们家宝宝刚出生","还没小人呢","小孩刚参加中考","一岁的小朋友没有","我刚上班，没有小孩","小孩还没到那个阶段呢","我们家小孩都上小学啦","没有没有，没有小孩","我们家没有一岁的娃","我都五十多了，哪来一岁的小孩","我们家小朋友刚出生","还没准备生小人","没有这么大的孩子","我们家崽都上幼儿园啦","我们家没有这么大的小孩","娃还在肚子里呢","刚结婚，还没打算要小孩","我女儿都十几岁了","我还在读书呢，没有小孩","宝宝还没到一岁","我们家女儿刚参加高考","小朋友还没有呢","小孩还没有这么大","我刚毕业，哪来的小孩","我家女儿都上初中了","我是单身，没有小孩","我儿子都上大学了"],"口碑差_7a8Ax7": ["你们骗人吧","不要冒充银行骗人好吧","你说你是不是骗子吧","你干嘛咯你是谁咯骗子吧","你们想骗我钱吧","你骗人吧","你骗来骗去有意思吗","电话诈骗要坐牢","你不要诈骗了","你骗来钱敢用吗","你骗人钱有命花吗","呃我这手机上怎么显示你诈骗","不会是诈骗电话吧","你骗来骗去以后要坐牢","你个骗子想骗我没那么容易","你不要骗我","不知道行了好不用应该都是诈骗哪个贷款","忽悠人骗人","行了行了行了唉呦不要干这种骗人勾当","你都是骗人我那个叫","骗人的吧","你到处骗人有什么好处","现在骗子这么多我凭什么相信你啊","你们骗子公司还是流氓公司啊","你骗钱要下地狱","你别骗人了吧","不知道九五九五二开头都是骗子","你不会是骗子吧","你不要骗人","那诈骗你这是类似诈骗电话","靠谱吗","你不要到处骗人今天你没有好下场","你们不会是骗子吧","嗯上介绍我骗","你骗人不好","电信诈骗吧","你告诉我你骗我有什么好处","嗯反正那上写一是诈骗","你这是诈骗电话吧","那个骗子","你你说你要骗人"]}}' http://172.16.113.103:1472/hnsw/update

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"model_name": "BertClsModel","org": "学历_0OVBb7","slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true}],"kps": {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"],"没小孩-幼儿3-8岁_EAdQu1": ["我都没有对象，怎么会有小孩","我儿子都二十多岁了","我们家宝宝三岁了","我们家宝宝刚出生","还没小人呢","小孩刚参加中考","一岁的小朋友没有","我刚上班，没有小孩","小孩还没到那个阶段呢","我们家小孩都上小学啦","没有没有，没有小孩","我们家没有一岁的娃","我都五十多了，哪来一岁的小孩","我们家小朋友刚出生","还没准备生小人","没有这么大的孩子","我们家崽都上幼儿园啦","我们家没有这么大的小孩","娃还在肚子里呢","刚结婚，还没打算要小孩","我女儿都十几岁了","我还在读书呢，没有小孩","宝宝还没到一岁","我们家女儿刚参加高考","小朋友还没有呢","小孩还没有这么大","我刚毕业，哪来的小孩","我家女儿都上初中了","我是单身，没有小孩","我儿子都上大学了"],"口碑差_7a8Ax7": ["你们骗人吧","不要冒充银行骗人好吧","你说你是不是骗子吧","你干嘛咯你是谁咯骗子吧","你们想骗我钱吧","你骗人吧","你骗来骗去有意思吗","电话诈骗要坐牢","你不要诈骗了","你骗来钱敢用吗","你骗人钱有命花吗","呃我这手机上怎么显示你诈骗","不会是诈骗电话吧","你骗来骗去以后要坐牢","你个骗子想骗我没那么容易","你不要骗我","不知道行了好不用应该都是诈骗哪个贷款","忽悠人骗人","行了行了行了唉呦不要干这种骗人勾当","你都是骗人我那个叫","骗人的吧","你到处骗人有什么好处","现在骗子这么多我凭什么相信你啊","你们骗子公司还是流氓公司啊","你骗钱要下地狱","你别骗人了吧","不知道九五九五二开头都是骗子","你不会是骗子吧","你不要骗人","那诈骗你这是类似诈骗电话","靠谱吗","你不要到处骗人今天你没有好下场","你们不会是骗子吧","嗯上介绍我骗","你骗人不好","电信诈骗吧","你告诉我你骗我有什么好处","嗯反正那上写一是诈骗","你这是诈骗电话吧","那个骗子","你你说你要骗人"]}}' http://172.16.113.103:1472/hnsw/update
@app.route('/hnsw/update',methods=['GET','POST'])
def hnsw_update():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        res = json_data['kps'] # 父亲节点对应所有子节点资源内容
        parent_node_name = json_data["org"] # 父亲节点
        index.hnsw_index_update(res, hnsw_model, parent_node_name)
        return res

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org":"开场白_yB7b2E","query": "骗人的吧"}' http://172.16.113.103:1472/hnsw/search
@app.route('/hnsw/search',methods=['GET','POST'])
def hnsw_search():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        parent_node_name = json_data['org'] # 父节点名称："开场白_yB7b2E"
        query = json_data['query'] # 用户query:骗人的吧
        print("query:{}".format(query))
        res = index.search(parent_node_name,query,2)
        res = res.to_json(force_ascii=False) #解决DataFrame中文to_json乱码 
        return res

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org": {"开场白_yB7b2E": {"use_model_for_token": 0,"use_protocol": 0,"rule_thres": 0.1,"resources": {"synonym": [],"protocol": [],"high_weight": [],"rule": [{"kp": "口碑差_7a8Ax7","rule": ["(口碑差|骗人|诈骗|骗子|骗我|欺骗|骗来骗去|骗来|诈骗)&~(靠不)/15"],"priority": []},{"kp": "不需要_EAdQu1","rule": ["(申请|需要|要贷款|办理)&(干嘛|干什么)&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间)/25","(没说|没有|没)&(申请|需要|要贷款|办理|需求)&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间|没通过|什么没有|啥子没有|啥玩意)/25","(不需要|不想要|没想要|没想用|不是很需要|不太需要|没得需要|没有需要|用不到|不考虑|没考虑|不贷款|不想贷|不想办|不想用|不感兴趣|没兴趣|不办|不做|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不太用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|不准备办|不太想办)&~(骗人|不要老是打电话|怎么借|打电话|打扰|骚扰|忽悠|骗钱|骗子|年纪大|年龄大|申请过|不贷给我|不要还|不用还|不需要还|亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万|打来|借给你|打了|不会弄|百万|抵押|申不申|我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼|不要说)/30"],"priority": []},{"kp": "黑户_oRqvs2","rule": ["(黑户|黑名单)/15"],"priority": []}],"slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1}],"same_class": [],"token": []},"use_rule": 1,"global_token_module": 0,"use_rule_for_token": 0}}}' http://172.16.113.103:1472/rule/update
# 更新规则接口
@app.route('/rule/update',methods=['GET','POST'])
def rule_update():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        res = parse_rdg_interface(json_data)
        print("req:{},res:{}".format(req,res))
        rengine.rules_update(res)
        return res

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org":"开场白_yB7b2E","query": "我是黑户"}' http://172.16.113.103:1472/rule/search
# 规则查询接口
@app.route('/rule/search',methods=['GET','POST'])
def rule_search():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        query = json_data['query']
        parent_node = json_data['org']
        res = rengine.rule_parse(parent_node,query)
        print("req:{},res:{}".format(req,res))
        return res.to_json(force_ascii=False) #解决DataFrame中文to_json乱码

# 解析研究院请求body
def parse_rdg_interface(json_data):
        parent_dict = {}
        for key,value in json_data['org'].items():
            parent_node_name = key #父节点名
            parent_node_resources = value['resources']
            son_node_rule = parent_node_resources['rule'] #子节点资源
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
            parent_dict[parent_node_name] = son_dict
        return parent_dict

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"query": "我姓汤喝汤的汤","org": [{ "org_name": "开场白_yB7b2E","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}' http://172.16.113.103:1472/nlu/search
@app.route('/nlu/search',methods=['GET','POST'])
def search():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        parent_node_name = json_data['org'][0]['org_name'] # 父节点名称："开场白_yB7b2E"
        query = json_data['query'] # 用户query:骗人的吧
        print("query:{}".format(query))
        res = Auto_Sort_Keyword_Semantic_V21(parent_node_name,query)
        return res

if __name__ == '__main__':
   app.config['JSON_AS_ASCII'] = False
   app.run(host='0.0.0.0',port=1472,debug=True) #运行开始 #阿里云机器需要申请1472端口防火墙开启

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"model_name": "BertClsModel","org": "开场白_yB7b2E","slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true}],"kps": {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"],"没小孩-幼儿3-8岁_EAdQu1": ["我都没有对象，怎么会有小孩","我儿子都二十多岁了","我们家宝宝三岁了","我们家宝宝刚出生","还没小人呢","小孩刚参加中考","一岁的小朋友没有","我刚上班，没有小孩","小孩还没到那个阶段呢","我们家小孩都上小学啦","没有没有，没有小孩","我们家没有一岁的娃","我都五十多了，哪来一岁的小孩","我们家小朋友刚出生","还没准备生小人","没有这么大的孩子","我们家崽都上幼儿园啦","我们家没有这么大的小孩","娃还在肚子里呢","刚结婚，还没打算要小孩","我女儿都十几岁了","我还在读书呢，没有小孩","宝宝还没到一岁","我们家女儿刚参加高考","小朋友还没有呢","小孩还没有这么大","我刚毕业，哪来的小孩","我家女儿都上初中了","我是单身，没有小孩","我儿子都上大学了"],"口碑差_7a8Ax7": ["你们骗人吧","不要冒充银行骗人好吧","你说你是不是骗子吧","你干嘛咯你是谁咯骗子吧","你们想骗我钱吧","你骗人吧","你骗来骗去有意思吗","电话诈骗要坐牢","你不要诈骗了","你骗来钱敢用吗","你骗人钱有命花吗","呃我这手机上怎么显示你诈骗","不会是诈骗电话吧","你骗来骗去以后要坐牢","你个骗子想骗我没那么容易","你不要骗我","不知道行了好不用应该都是诈骗哪个贷款","忽悠人骗人","行了行了行了唉呦不要干这种骗人勾当","你都是骗人我那个叫","骗人的吧","你到处骗人有什么好处","现在骗子这么多我凭什么相信你啊","你们骗子公司还是流氓公司啊","你骗钱要下地狱","你别骗人了吧","不知道九五九五二开头都是骗子","你不会是骗子吧","你不要骗人","那诈骗你这是类似诈骗电话","靠谱吗","你不要到处骗人今天你没有好下场","你们不会是骗子吧","嗯上介绍我骗","你骗人不好","电信诈骗吧","你告诉我你骗我有什么好处","嗯反正那上写一是诈骗","你这是诈骗电话吧","那个骗子","你你说你要骗人"]}}' http://172.16.113.103:1472/hnsw/update

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org":"开场白_yB7b2E","query": "骗人的吧"}' http://172.16.113.103:1472/hnsw/search



#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"model_name": "BertClsModel","org": "开场白_yB7b2E","slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true}],"kps": {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"],"没小孩-幼儿3-8岁_EAdQu1": ["我都没有对象，怎么会有小孩","我儿子都二十多岁了","我们家宝宝三岁了","我们家宝宝刚出生","还没小人呢","小孩刚参加中考","一岁的小朋友没有","我刚上班，没有小孩","小孩还没到那个阶段呢","我们家小孩都上小学啦","没有没有，没有小孩","我们家没有一岁的娃","我都五十多了，哪来一岁的小孩","我们家小朋友刚出生","还没准备生小人","没有这么大的孩子","我们家崽都上幼儿园啦","我们家没有这么大的小孩","娃还在肚子里呢","刚结婚，还没打算要小孩","我女儿都十几岁了","我还在读书呢，没有小孩","宝宝还没到一岁","我们家女儿刚参加高考","小朋友还没有呢","小孩还没有这么大","我刚毕业，哪来的小孩","我家女儿都上初中了","我是单身，没有小孩","我儿子都上大学了"],"口碑差_7a8Ax7": ["你们骗人吧","不要冒充银行骗人好吧","你说你是不是骗子吧","你干嘛咯你是谁咯骗子吧","你们想骗我钱吧","你骗人吧","你骗来骗去有意思吗","电话诈骗要坐牢","你不要诈骗了","你骗来钱敢用吗","你骗人钱有命花吗","呃我这手机上怎么显示你诈骗","不会是诈骗电话吧","你骗来骗去以后要坐牢","你个骗子想骗我没那么容易","你不要骗我","不知道行了好不用应该都是诈骗哪个贷款","忽悠人骗人","行了行了行了唉呦不要干这种骗人勾当","你都是骗人我那个叫","骗人的吧","你到处骗人有什么好处","现在骗子这么多我凭什么相信你啊","你们骗子公司还是流氓公司啊","你骗钱要下地狱","你别骗人了吧","不知道九五九五二开头都是骗子","你不会是骗子吧","你不要骗人","那诈骗你这是类似诈骗电话","靠谱吗","你不要到处骗人今天你没有好下场","你们不会是骗子吧","嗯上介绍我骗","你骗人不好","电信诈骗吧","你告诉我你骗我有什么好处","嗯反正那上写一是诈骗","你这是诈骗电话吧","那个骗子","你你说你要骗人"]}}' http://172.21.191.94:1472/hnsw/update

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"model_name": "BertClsModel","org": "开场白_yB7b2E","slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true}],"kps": {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"]}}' http://172.21.191.94:1472/hnsw/update

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"query": "我姓汤喝汤的汤"}' http://172.21.191.94:1472/hnsw/search

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"query": "骗人的吧"}' http://172.21.191.94:1472/hnsw/search

#curl -H "Content-Type: application/x-www-form-urlencoded" -X POST  --data '{"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"]}' http://172.21.191.94:1472/hnsw/update