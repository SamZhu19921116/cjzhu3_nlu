# -*- coding: utf-8 -*-
import os
import json
import time
import args
import torch
import logging
import numpy as np
import pandas as pd
from flask import Flask, request
from RULE_ENGINE import Rule_Engine
from SLOT_ENGINE import Slot_Engine
from HNSW_SBERT_RECALL_ENGINE import HNSW_Recall
from BERT_NER.models.bert_for_ner import BertSoftmaxForNer
from BERT_NER.processors.ner_seq import ner_processors
from transformers import BertTokenizer

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

from jieba import Tokenizer
# os.remove("/tmp/jieba.cache") #每次删除结巴缓存
jieba_model = Tokenizer("/usr/local/anaconda3/envs/nlu/lib/python3.7/site-packages/jieba/dict.txt")
from gensim import models
w2v_model = models.KeyedVectors.load(args.sort_word2vec_model_zhongan)

############################Keyword_Semantic模块###############################
# 加载双塔BERT预训练模型
from sentence_transformers import SentenceTransformer, util
sbert_model = SentenceTransformer(args.retrival_sbert)
# 基于多种度量方式的计算两句子的相似度
from Sort_Similarity_Calculater import Similarity_Measure
sim_calc = Similarity_Measure()
# 双塔BERT文本相似度度量
from Sentence_Bert_Calculater import SBertSimCalc
sbert_sim = SBertSimCalc(sbert_model=sbert_model)

hnsw_model = os.path.join(os.path.dirname(__file__),"hnsw_model")
hnsw_engine = HNSW_Recall(sbert_model, w2v_model, jieba_model, hnsw_model)
rule_model = os.path.join(os.path.dirname(__file__),"rules_model")
rule_engine = Rule_Engine(rule_model)

def cos_sim(nd_a, nd_b):
    nd_a = np.array(nd_a)
    nd_b = np.array(nd_b)
    return np.sum(nd_a * nd_b) / (np.sqrt(np.sum(nd_a**2)) * np.sqrt(np.sum(nd_b**2)))

import difflib
def Difflib_SeqMatcher(str_a , str_b):
    return difflib.SequenceMatcher(None, str_a, str_b).ratio()

# 召回topN问题及精排序(按节点分层，每层分不同召回集)
# @profile
def Recall_Sort_V1(parent_node_name,query,topN = 10):
    # query问题粗召回
    RoughRecall_DictList = hnsw_engine.recall(parent_node_name, query, topN)
    RoughRecall_ListDict = [dict(zip(tuple(RoughRecall_DictList.keys()),t)) for t in list(zip(*(RoughRecall_DictList.values())))]
    sim = {}
    RecallSort = []
    for row in RoughRecall_ListDict:
        sim['query'] = row['query']
        sim['retrieved'] = row['recall_sentences']
        sim['intent'] = row['recall_intents']
        sim['w2v_cos'] = cos_sim(row['query_word2vec'],row['recall_word2vec'])
        sim['diff_score'] = Difflib_SeqMatcher(row['query'],row['recall_sentences'])
        sim['sbert_score'] = util.cos_sim(row['query_embedding'], row['recall_embedding']).numpy()[0][0]
        RecallSort.append(sim)
    # {'query','retrieved','intent','lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','sbert_score'}
    return RecallSort

# @profile
def Auto_Sort_Keyword_Semantic_V11(parent_node, query, model_threshold = 0.99):
    FineSortRes = None
    FineSortList = []
    FineSortRule = rule_engine.rules_parse(parent_node,query)
    RecallSort = Recall_Sort_V1(parent_node_name=parent_node,query=query)
    FineSortSbert = sorted(RecallSort, key=lambda x:x['sbert_score'], reverse = True)[0]
    FineSortList.append({"match_score":FineSortSbert['sbert_score'],"match_node":FineSortSbert['intent'],"match_rule":FineSortSbert['retrieved'],"match_module":"sbert"})
    FineSortW2V = sorted(RecallSort, key=lambda x:x['w2v_cos'], reverse = True)[0]
    FineSortList.append({"match_score":FineSortW2V['w2v_cos'],"match_node":FineSortW2V['intent'],"match_rule":FineSortW2V['retrieved'],"match_module":"word2vec"})
    FineSortDiff = sorted(RecallSort, key=lambda x:x['diff_score'], reverse = True)[0]
    FineSortList.append({"match_score":FineSortDiff['diff_score'],"match_node":FineSortDiff['intent'],"match_rule":FineSortDiff['retrieved'],"match_module":"difflib"})

    if len(FineSortList) != 0:
        FineSortRes = sorted(FineSortList, key=lambda x:x['match_score'], reverse = True)[0]
        FineSortList.append(FineSortRule)
    else:
        FineSortList.append(FineSortRule)
        FineSortRes = sorted(FineSortList, key=lambda x:x['match_score'], reverse = True)[0]
    # {"match_score", "match_node", "match_rule", "match_module"}
    predict_intent = {"module":FineSortRes["match_module"], "rule":FineSortRes["match_rule"], "kp":FineSortRes["match_node"], "score":str(FineSortRes["match_score"])}
    co_exist_intents = [{"kp":item['match_node'], "module":item['match_module'], "rule":item['match_rule'], "score":str(item['match_score'])} for item in FineSortList]
    return predict_intent, co_exist_intents

# 召回topN问题及精排序(按节点分层，每层分不同召回集)
# @profile
def Fine_Sort_V2(parent_node_name,query,topN = 10):
    # query问题粗召回
    # {'query':query_list,'query_embedding':query_embedding_list,'recall_sentences':recall_sentences_list, 'recall_intents':recall_intents_list, 'recall_distances':recall_distances_list, 'recall_embedding':recall_embedding_list}
    RoughRecallDF = hnsw_engine.search(parent_node_name, query, topN)
    # 多种度量方式计算query问题以及query召回问题的相似度
    FineSort = pd.DataFrame.from_records(RoughRecallDF.apply(lambda row: sim_calc.Similarity_Calculate_V2(row['query'], row['recall_sentences'], row['recall_intents']), axis=1))
    FineSort['sbert_score'] = RoughRecallDF.apply(lambda row: util.cos_sim(row['query_embedding'], row['recall_embedding']).numpy()[0][0] , axis=1)
    return FineSort

# @profile
def Auto_Sort_Keyword_Semantic_V21(parent_node,query,model_threshold = 0.99):
    # {"score":score_max,"hit_node":matcher_son_node,"match_query":matcher_son_rule}
    df_list = []
    result = None
    df_tmp = None
    # 规则类的匹配DataFrame:[score,hit_node,match_query]
    RuleParse = rule_engine.parse(parent_node,query)
    RuleSort = RuleParse.head(1)
    # 粗召回+精排序DataFrame:[score,hit_node,match_query]
    FineSort = Fine_Sort_V2(parent_node_name=parent_node,query=query)
    # SBERT
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"match_score":FineSortSbert.iloc[0,-1],"match_node":FineSortSbert.iloc[0,-2],"match_rule":FineSortSbert.iloc[0,-3],"match_module":"sbert"},index = [1]))
    # WORD2VEC
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"match_score":FineSortW2V.iloc[0,-1],"match_node":FineSortW2V.iloc[0,-2],"match_rule":FineSortW2V.iloc[0,-3],"match_module":"word2vec"},index = [2]))
    # DIFFLIB
    FineSortDiff = FineSort[['query','retrieved', 'intent','diff_score']].sort_values(by=['diff_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"match_score":FineSortDiff.iloc[0,-1],"match_node":FineSortDiff.iloc[0,-2],"match_rule":FineSortDiff.iloc[0,-3],"match_module":"difflib"},index = [3]))
    
    df_tmp = pd.concat(df_list,ignore_index=True)
    result = df_tmp.loc[lambda x:x['match_score'] > model_threshold]
    if not result.empty:
        result = result.sort_values(by=['match_score'],ascending=False).head(1)
        df_list.append(RuleSort)
        df_tmp = pd.concat(df_list,ignore_index=True)
    else:
        df_list.append(RuleSort)
        df_tmp = pd.concat(df_list, ignore_index=True)
        result = df_tmp.sort_values(by=['match_score'],ascending=False).head(1)

    #{精排命中模块,精排命中关键词集,精排命中节点,精排最终得分}
    predict_intent = {"module":result.iloc[0,-1], "rule":result.iloc[0,-2], "kp":result.iloc[0,-3], "score":result.iloc[0,-4]} 
    co_exist_intents = [{"kp":row['match_node'], "module":row['match_module'], "rule":row['match_rule'], "score":row['match_score']} for index, row in df_tmp.iterrows()]
    return predict_intent, co_exist_intents

@app.route('/test')
def test():
    return '<h1>Hello World</h1>'

# curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"model_name": "BertClsModel","org": "开场白_yB7b2E","slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true}],"kps": {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"],"没小孩-幼儿3-8岁_EAdQu1": ["我都没有对象，怎么会有小孩","我儿子都二十多岁了","我们家宝宝三岁了","我们家宝宝刚出生","还没小人呢","小孩刚参加中考","一岁的小朋友没有","我刚上班，没有小孩","小孩还没到那个阶段呢","我们家小孩都上小学啦","没有没有，没有小孩","我们家没有一岁的娃","我都五十多了，哪来一岁的小孩","我们家小朋友刚出生","还没准备生小人","没有这么大的孩子","我们家崽都上幼儿园啦","我们家没有这么大的小孩","娃还在肚子里呢","刚结婚，还没打算要小孩","我女儿都十几岁了","我还在读书呢，没有小孩","宝宝还没到一岁","我们家女儿刚参加高考","小朋友还没有呢","小孩还没有这么大","我刚毕业，哪来的小孩","我家女儿都上初中了","我是单身，没有小孩","我儿子都上大学了"],"口碑差_7a8Ax7": ["你们骗人吧","不要冒充银行骗人好吧","你说你是不是骗子吧","你干嘛咯你是谁咯骗子吧","你们想骗我钱吧","你骗人吧","你骗来骗去有意思吗","电话诈骗要坐牢","你不要诈骗了","你骗来钱敢用吗","你骗人钱有命花吗","呃我这手机上怎么显示你诈骗","不会是诈骗电话吧","你骗来骗去以后要坐牢","你个骗子想骗我没那么容易","你不要骗我","不知道行了好不用应该都是诈骗哪个贷款","忽悠人骗人","行了行了行了唉呦不要干这种骗人勾当","你都是骗人我那个叫","骗人的吧","你到处骗人有什么好处","现在骗子这么多我凭什么相信你啊","你们骗子公司还是流氓公司啊","你骗钱要下地狱","你别骗人了吧","不知道九五九五二开头都是骗子","你不会是骗子吧","你不要骗人","那诈骗你这是类似诈骗电话","靠谱吗","你不要到处骗人今天你没有好下场","你们不会是骗子吧","嗯上介绍我骗","你骗人不好","电信诈骗吧","你告诉我你骗我有什么好处","嗯反正那上写一是诈骗","你这是诈骗电话吧","那个骗子","你你说你要骗人"]}}' http://172.16.113.103:1472/hnsw/update

# curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"model_name": "BertClsModel","org": "学历_0OVBb7","slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": false,"if_use_model": true,"if_use_rule": true}],"kps": {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"],"没小孩-幼儿3-8岁_EAdQu1": ["我都没有对象，怎么会有小孩","我儿子都二十多岁了","我们家宝宝三岁了","我们家宝宝刚出生","还没小人呢","小孩刚参加中考","一岁的小朋友没有","我刚上班，没有小孩","小孩还没到那个阶段呢","我们家小孩都上小学啦","没有没有，没有小孩","我们家没有一岁的娃","我都五十多了，哪来一岁的小孩","我们家小朋友刚出生","还没准备生小人","没有这么大的孩子","我们家崽都上幼儿园啦","我们家没有这么大的小孩","娃还在肚子里呢","刚结婚，还没打算要小孩","我女儿都十几岁了","我还在读书呢，没有小孩","宝宝还没到一岁","我们家女儿刚参加高考","小朋友还没有呢","小孩还没有这么大","我刚毕业，哪来的小孩","我家女儿都上初中了","我是单身，没有小孩","我儿子都上大学了"],"口碑差_7a8Ax7": ["你们骗人吧","不要冒充银行骗人好吧","你说你是不是骗子吧","你干嘛咯你是谁咯骗子吧","你们想骗我钱吧","你骗人吧","你骗来骗去有意思吗","电话诈骗要坐牢","你不要诈骗了","你骗来钱敢用吗","你骗人钱有命花吗","呃我这手机上怎么显示你诈骗","不会是诈骗电话吧","你骗来骗去以后要坐牢","你个骗子想骗我没那么容易","你不要骗我","不知道行了好不用应该都是诈骗哪个贷款","忽悠人骗人","行了行了行了唉呦不要干这种骗人勾当","你都是骗人我那个叫","骗人的吧","你到处骗人有什么好处","现在骗子这么多我凭什么相信你啊","你们骗子公司还是流氓公司啊","你骗钱要下地狱","你别骗人了吧","不知道九五九五二开头都是骗子","你不会是骗子吧","你不要骗人","那诈骗你这是类似诈骗电话","靠谱吗","你不要到处骗人今天你没有好下场","你们不会是骗子吧","嗯上介绍我骗","你骗人不好","电信诈骗吧","你告诉我你骗我有什么好处","嗯反正那上写一是诈骗","你这是诈骗电话吧","那个骗子","你你说你要骗人"]}}' http://172.16.113.103:1472/hnsw/update
@app.route('/hnsw/update',methods=['GET','POST'])
def hnsw_update():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        parent_node_name = json_data['org'] # 父亲节点
        slot_map = json_data['slot_map']
        # 将语料中申请的槽值提取字典保存至指定文件中:slot_model/parent_node_name.pkl(此项功能仅为适配RDG接口，暂时仅开放语料中槽值提取功能)
        slot_engine.slot_update(slot_map, slot_model, parent_node_name)
        res = json_data['kps'] # 父亲节点对应所有子节点资源内容
        hnsw_engine.hnsw_index_update(res, hnsw_model, parent_node_name)
        return res

# curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org":"开场白_yB7b2E","query": "骗人的吧"}' http://172.16.113.103:1472/hnsw/search
@app.route('/hnsw/search',methods=['GET','POST'])
def hnsw_search():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        parent_node_name = json_data['org'] # 父节点名称："开场白_yB7b2E"
        query = json_data['query'] # 用户query:骗人的吧
        print("query:{}".format(query))
        res = hnsw_engine.search(parent_node_name,query,2)
        res = res.to_json(force_ascii=False) #解决DataFrame中文to_json乱码 
        return res

# curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org": {"开场白_yB7b2E": {"use_model_for_token": 0,"use_protocol": 0,"rule_thres": 0.1,"resources": {"synonym": [],"protocol": [],"high_weight": [],"rule": [{"kp": "口碑差_7a8Ax7","rule": ["(口碑差|骗人|诈骗|骗子|骗我|欺骗|骗来骗去|骗来|诈骗)&~(靠不)/15"],"priority": []},{"kp": "不需要_EAdQu1","rule": ["(申请|需要|要贷款|办理)&(干嘛|干什么)&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间)/25","(没说|没有|没)&(申请|需要|要贷款|办理|需求)&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间|没通过|什么没有|啥子没有|啥玩意)/25","(不需要|不想要|没想要|没想用|不是很需要|不太需要|没得需要|没有需要|用不到|不考虑|没考虑|不贷款|不想贷|不想办|不想用|不感兴趣|没兴趣|不办|不做|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不太用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|不准备办|不太想办)&~(骗人|不要老是打电话|怎么借|打电话|打扰|骚扰|忽悠|骗钱|骗子|年纪大|年龄大|申请过|不贷给我|不要还|不用还|不需要还|亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万|打来|借给你|打了|不会弄|百万|抵押|申不申|我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼|不要说)/30"],"priority": []},{"kp": "黑户_oRqvs2","rule": ["(黑户|黑名单)/15"],"priority": []}],"slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1}],"same_class": [],"token": []},"use_rule": 1,"global_token_module": 0,"use_rule_for_token": 0}}}' http://172.16.113.103:1472/rule/update
# 更新规则接口
@app.route('/rule/update',methods=['GET','POST'])
def rule_update():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        res,slot_map = parse_rdg_interface(json_data)
        # 将规则中申请的槽值提取字典保存至指定文件中:slot_model/parent_node_name.pkl(此项功能仅为适配RDG接口，暂时不开放)
        # slot_engine.slot_update(slot_map, slot_model, parent_node_name)
        for parent_node_name,resource in res.items():
            rule_engine.rules_update(resource,rule_model,parent_node_name)
        return res

# curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org":"开场白_yB7b2E","query": "我是黑户"}' http://172.16.113.103:1472/rule/search
# curl http://172.16.113.103:1472/
# curl http://120.26.167.21:1472/
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
            parent_node_name = key # 父节点名
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
            parent_dict[parent_node_name] = son_dict
        return parent_dict, slot_map

# curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"query": "我姓汤喝汤的汤","org": [{ "org_name": "开场白_yB7b2E","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}' http://172.16.113.103:1472/nlu/search
@app.route('/nlu/search',methods=['GET','POST'])
def search():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        parent_node_name = json_data['org'][0]['org_name'] # 父节点名称："开场白_yB7b2E"
        query = json_data['query'] # 用户query:骗人的吧
        sem_res,co_sem_res = None,None
        try:
            #sem_res,co_sem_res = Auto_Sort_Keyword_Semantic_V21(parent_node_name,query)
            sem_res,co_sem_res = Auto_Sort_Keyword_Semantic_V11(parent_node_name,query)
            logging.info("org:{},query:{},predict_intent:{}".format(parent_node_name,query,sem_res))
        except:
            logging.info("输入节点:{},输入文本:{}".format(parent_node_name,query))
        
        slot_dict = slot_engine.slot_module_dict
        ner_res = {}
        try:
            for item in slot_dict[parent_node_name]:
                kp = item['kp']
                slot_name = item['slot_name']
                if_use_model = item['if_use_model']
                if_use_rule = item['if_use_rule']
                if if_use_model and kp == sem_res['kp']:
                    ner_res = Ner_Predict(query,model,tokenizer)
                    sem_res["slot"]=ner_res
        except:
            logging.info("输入节点错误2:{}".format(parent_node_name))
            
        result = {"module":"cls","org":parent_node_name,"predict_intent":sem_res,"query":query,"recommend":[],"slot":ner_res,"emotion":{},"co_exist_intents":co_sem_res}
        return {"result":result,"status":0}

if __name__ == '__main__':
    # start = time.time()
    # sem_res,co_sem_res = Auto_Sort_Keyword_Semantic_V11('开场白_yB7b2E', '我不要')
    # end = time.time()
    # print("The function run time is : %.03f seconds" %((end-start)/100))
    # start = time.time()
    # for i in range(100):
    #     sem_res,co_sem_res = Auto_Sort_Keyword_Semantic_V21('开场白', '我不要')
    # end = time.time()
    # print("The function run time is : %.03f seconds" %((end-start)/100))
    app.run(host='0.0.0.0',port=1472,debug=True) # 运行开始 #阿里云机器需要申请1472端口防火墙开启
    # 内网地址：http://172.16.113.103:1472/
    # 公网地址：http://120.26.167.21:1472/

    # ps -ef | grep python | grep -v grep | awk '{print $2}' | xargs kill -9
    # curl -X POST http://120.26.167.21:1472/nlu/search -d '{"query": "你不要诈骗了","org": [{"org_name": "开场白","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}'
    # curl -X POST http://localhost:1472/nlu/search -d '{"query": "你不要诈骗了","org": [{"org_name": "开场白","cls_score": 0.7,"rule_score": 0.7,"intent_list": []}]}'
    # ./wrk -t 16 -c 500 -d 30s --latency --timeout 10s -s post.lua http://localhost:1472/nlu/search
    # ./wrk -t 5 -c 100 -d 30s --latency --timeout 5s -s post.lua http://120.26.167.21:1472/nlu/search
    # ./wrk -t 5 -c 100 -d 30s --latency -s post.lua http://120.26.167.21:1472/nlu/search
    # 加--preload 可以查到代码具体错误
    # gunicorn -c gunicorn.py nlu_server_v202201061:app --preload
    # ps ax | grep gunicorn
    # pstree -ap | grep gunicorn : 获取 Gunicorn 进程树
    # kill -9 24810 : 彻底杀死 Gunicorn 服务
    # kill -HUP 24810 : 重启 Gunicorn 服务

    # ./wrk -t 16 -c 100 -d 30s --latency --timeout 5s -s post.lua http://localhost:1472/nlu/search
    # Running 30s test @ http://localhost:1472/nlu/search
    # 16 threads and 100 connections
    # Thread Stats   Avg      Stdev     Max   +/- Stdev
    #     Latency     4.44s   127.27ms   4.53s   100.00%
    #     Req/Sec     1.61      2.62    10.00     83.33%
    # Latency Distribution
    #     50%    4.53s 
    #     75%    4.53s 
    #     90%    4.53s 
    #     99%    4.53s 
    # 18 requests in 30.04s, 33.36KB read
    # Socket errors: connect 0, read 0, write 0, timeout 16
    # Requests/sec:      0.60
    # Transfer/sec:      1.11KB

    # kernprof -l -v jieba_cut_test.py >jieba_cut_test.log
    # kernprof -l -v nlu_server_v202201061.py >kernprof.log