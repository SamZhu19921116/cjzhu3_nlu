import logging
import os
from urllib import response
import args
import json
from flask import Flask, request

app=Flask(__name__) # 创建新的开始
app.config['JSON_AS_ASCII'] = False # 解决中文乱码问题

import jieba
if os.path.exists("/tmp/jieba.cache"):
    os.remove("/tmp/jieba.cache") #每次删除结巴缓存
jieba.load_userdict("/usr/local/anaconda3/envs/nlu/lib/python3.7/site-packages/jieba/dict.txt")

from gensim import models
w2v_model = models.KeyedVectors.load(args.sort_word2vec_model_zhongan)

from sentence_transformers import SentenceTransformer, util
from HNSW_SBERT_RECALL_ENGINE import HNSW_Recall
sbert_model = SentenceTransformer(args.retrival_sbert)
hnsw_model = os.path.join(os.path.dirname(__file__),"hnsw_model")
hnsw_engine = HNSW_Recall(sbert_model, w2v_model, jieba, hnsw_model)

import numpy as np
def cos_sim(nd_a, nd_b):
    nd_a = np.array(nd_a)
    nd_b = np.array(nd_b)
    return np.sum(nd_a * nd_b) / (np.sqrt(np.sum(nd_a**2)) * np.sqrt(np.sum(nd_b**2)))

import difflib
def Difflib_SeqMatcher(str_a , str_b):
    return difflib.SequenceMatcher(None, str_a, str_b).ratio()

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
        sim['sbert_score'] = util.cos_sim(row['query_embedding'], row['recall_embedding']).numpy()[0][0].tolist()
        RecallSort.append(sim)
    # {'query','retrieved','intent','lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','sbert_score'}
    return RecallSort

@app.route('/recall',methods=['GET','POST'])
def rough_recall():
    if request.method == 'POST':
        data = request.get_data(as_text=True)
        json_data = json.loads(data)
        parent_node_name = json_data['org'] # 父节点名称："开场白_yB7b2E"
        query = json_data['query'] # 用户query:骗人的吧
        model_recall = Recall_Sort_V1(parent_node_name, query, 10)
        response = {"rough_recall":model_recall}
        logging.info("req:{},res:{}".format(data,response))
        return response

if __name__ == "__main__":
    app.run(host='localhost',port=1472,debug=True) # 运行开始 #阿里云机器需要申请1472端口防火墙开启
    # curl -X POST http://localhost:1472/recall -d '{"query": "你不要诈骗了","org": "开场白"}'
    # ab -n50 -c1 -T application/json -p recall.json  http://127.0.0.1:1472/recall
    # gunicorn -c gunicorn.py nlu_recall_server:app
    # pstree -ap | grep gunicorn