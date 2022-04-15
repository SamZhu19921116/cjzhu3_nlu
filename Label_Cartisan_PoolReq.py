import itertools

import tokenization
from tqdm import tqdm
import requests
import difflib
import numpy as np
import json

tokenizer = tokenization.FullTokenizer(vocab_file='/data/jdduan/sophon_cbg_2.0.3/data/cls/vocab.txt', do_lower_case=True)

# 将中文文本id化用于请求bert模型
def convert_sentence_to_features(sentence,max_len = 32):
    tokens = []
    tokens.append("[CLS]")
    for token in tokenizer.tokenize(sentence):
        tokens.append(token)
    tokens.append("[SEP]")
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    while len(input_ids) < max_len:
        input_ids.append(0)
    return input_ids

# 计算语句向量的相似度
def cosine_sim(nd_a, nd_b):
    nd_a = np.array(nd_a)
    nd_b = np.array(nd_b)
    return np.sum(nd_a * nd_b) / (np.sqrt(np.sum(nd_a**2)) * np.sqrt(np.sum(nd_b**2)))

def seq_embed_req(*arg):
    info = arg[0]
    ids = info[0]
    index = info[1]
    embedding_tmp = requests.post("http://127.0.0.1:8501/v1/models/model_cls_slot:predict",data=json.dumps({"instances": [{"ori_input_quests": ids}]}))
    embedding_res = embedding_tmp.json()['predictions'][0]
    cls_seq_emb = np.array(embedding_res['cls_seq_emb'])
    return (cls_seq_emb,index)

def difflib_sim(str_a , str_b):
    return difflib.SequenceMatcher(None, str_a, str_b).ratio()

in_path = "/root/cjzhu3_nlu/data/CreditCard_ZhaiJi_Label.txt"
query_list = []
with open(in_path,mode="r",encoding='UTF-8') as inData:
    for line in inData:
        query_list.append(line.strip())

out_path0 = "/root/cjzhu3_nlu/data/CreditCard_ZhaiJi_Label_Pool_Same.txt"
fp0 = open(out_path0,mode="w+",encoding="utf8")

out_path1 = "/root/cjzhu3_nlu/data/CreditCard_ZhaiJi_Label_Pool_NotSame.txt"
fp1 = open(out_path1,mode="w+",encoding="utf8")

query_pair_list = []
item0_ids_list = []
item1_ids_list = []
index_query_dict = {}
index = 0
for item in tqdm(list(itertools.combinations(query_list,2))):
    if item[0] != item[1]:
        try:
            line0 = item[0].split("\t")
            line1 = item[1].split("\t")
            query0 = line0[0]
            label0 = line0[1]
            query1 = line1[0]
            label1 = line1[1]

            item0_ids = convert_sentence_to_features(query0)
            item1_ids = convert_sentence_to_features(query1)
            item0_ids_list.append((item0_ids,index))
            item1_ids_list.append((item1_ids,index))
        
            diff_score = difflib_sim(query0, query1)
            index_query_dict[index] = item[0]+"\t"+item[1]+"\t"+str(diff_score)

            index += 1
        except:
            pass

from multiprocessing import Pool
#进程池子
pool = Pool(10)
#偏函数多进程请求
result0 = []
result0_dict = {}
for item in tqdm(item0_ids_list):
    res = pool.apply_async(seq_embed_req,(item,))
    result0.append(res)
print("################请求一结束###############")
result1 = []
result1_dict = {}
for item in tqdm(item1_ids_list):
    res = pool.apply_async(seq_embed_req,(item,))
    result1.append(res)
print("################请求二结束###############")
pool.close()
pool.join
for res in tqdm(result0):
    try:
        item = res.get()
        result0_dict[item[1]] = item[0]
    except:
        pass

for res in tqdm(result1):
    try:
        item = res.get()
        result1_dict[item[1]] = item[0]
    except:
        pass

print(result0_dict)
print(result1_dict)
print("################写入开始###############")
for key,value in index_query_dict.items():
    lines = value.split("\t")
    query0 = lines[0]
    label0 = lines[1]
    query1 = lines[2]
    label1 = lines[3]
    diff_score = float(lines[4])
    try:
        item0_embed = result0_dict[key]
        item1_embed = result1_dict[key]
        sbert_score = cosine_sim(item0_embed,item1_embed)
        average_score = (diff_score + sbert_score) / 2
        if label0 == label1 and query0 != query1 and len(query0) > 5 and len(query1) > 5:
            fp0.write(query0+"\t"+label0+"\t"+query1+"\t"+label1+"\t"+str(diff_score)+"\t"+str(sbert_score)+"\t"+str(average_score)+"\n")
        elif label0 != label1 and query0 != query1 and len(query0) > 5 and len(query1) > 5:
            fp1.write(query0+"\t"+label0+"\t"+query1+"\t"+label1+"\t"+str(diff_score)+"\t"+str(sbert_score)+"\t"+str(average_score)+"\n")
    except:
        pass

fp0.close()
fp1.close()