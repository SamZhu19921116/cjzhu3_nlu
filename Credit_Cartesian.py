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

def seq_embed_req(ids):
    embedding_tmp = requests.post("http://127.0.0.1:8501/v1/models/model_cls_slot:predict",data=json.dumps({"instances": [{"ori_input_quests": ids}]}))
    embedding_res = embedding_tmp.json()['predictions'][0]
    cls_seq_emb = np.array(embedding_res['cls_seq_emb'])
    return cls_seq_emb

def difflib_sim(str_a , str_b):
    return difflib.SequenceMatcher(None, str_a, str_b).ratio()

in_path = "/root/cjzhu3_nlu/data/CreditCard_ZhaiJi_Corpus_1510.txt"
query_list = []
with open(in_path,mode="r",encoding='UTF-8') as inData:
    for line in inData:
        query_list.append(line.strip())

out_path = "/root/cjzhu3_nlu/data/CreditCard_ZhaiJi_Corpus_1510_Cartesian.txt"
fp = open(out_path,mode="w+",encoding="utf8")

query_pair_list = []
for item in tqdm(list(itertools.product(query_list,query_list))):
    if item[0] != item[1]:
        try:
            item0_ids = convert_sentence_to_features(item[0])
            item1_ids = convert_sentence_to_features(item[1])
            item0_embed = seq_embed_req(item0_ids)
            item1_embed = seq_embed_req(item1_ids)
            diff_score = difflib_sim(item[0], item[1])
            sbert_score = cosine_sim(item0_embed,item1_embed)
            fp.write(item[0] + "\t"+ item[1] + "\t"+ str(diff_score)  + "\t"+ str(sbert_score) + "\n")
            # print(item[0],item[1],diff_score,sbert_score)
        except:
            pass
fp.close()