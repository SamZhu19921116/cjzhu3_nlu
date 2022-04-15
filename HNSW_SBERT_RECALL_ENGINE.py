#coding=utf-8
import os
import re
import args
import pickle
import hnswlib
import logging
import numpy as np
import pandas as pd
from jieba import Tokenizer
import requests

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec,KeyedVectors
class HNSW_Recall(object):
    def __init__(self,sbert_model,wor2vec_model,jieba_model,hnsw_model_dir=None): # ef=args.retrival_hnsw_ef, M=args.retrival_hnsw_max,data_path=None, hnsw_sbert_model_path=args.retrival_sbert_hnsw_model
        # 加载sentence-bert模型
        if isinstance(sbert_model,SentenceTransformer):
            self.model = sbert_model
        elif isinstance(sbert_model,str):
            self.model = SentenceTransformer(sbert_model)
        else:
            raise TypeError('Input Type Error !')

        # 加载word2vec模型
        if isinstance(wor2vec_model,Word2Vec):
            self.word2vec = wor2vec_model
        elif isinstance(wor2vec_model,str):
            self.word2vec = KeyedVectors.load(wor2vec_model)
        else:
            raise TypeError('Input Type Error !')

        # 加载jieba_model模型
        # if isinstance(jieba_model,Tokenizer):
        #     self.jieba = jieba_model
        # elif isinstance(jieba_model,str):
        #     self.jieba = Tokenizer(jieba_model)
        # else:
        #     raise TypeError('Input Type Error !')
        self.jieba = jieba_model
        
        self.hnsw_index = {}
        self.id_sentence_intent_map = {} # {"parent_node_name":{"id_intent_dict":id_intent_dict,"id_sentence_dict":id_sentence_dict}}
        self.sentence_embedding_map = {}
        # 若模型已存在则加载图模型否则重新构建新图模型
        if os.path.isdir(hnsw_model_dir) and len(os.listdir(hnsw_model_dir)):
            file_name_list = os.listdir(hnsw_model_dir)
            # file_name = os.path.basename(hnsw_model).split(".")[0]
            for file_name in file_name_list:
                if file_name.endswith(".bin"):
                    self.hnsw_index[file_name.strip(".bin")] = self.load_hnsw_index(os.path.join(hnsw_model_dir,file_name))
                    logging.info("加载HNSW召回模型:{}".format(os.path.join(hnsw_model_dir,file_name))) 
                elif file_name.endswith(".pkl"):
                    self.id_sentence_intent_map[file_name.strip(".pkl")] = self.json_deserialization(os.path.join(hnsw_model_dir,file_name))
                    logging.info("加载HNSW召回模型映射关系:{}".format(os.path.join(hnsw_model_dir,file_name)))

    # 更新hnsw模型
    # kps: {"姓名_oRqvs2": ["姓汤喝汤的汤","免贵姓汤喝汤的汤","我叫汤建飞","我姓汤喝汤的汤","我是汤建飞"],"没小孩-幼儿3-8岁_EAdQu1": ["我都没有对象，怎么会有小孩","我儿子都二十多岁了","我们家宝宝三岁了","我们家宝宝刚出生","还没小人呢","小孩刚参加中考","一岁的小朋友没有","我刚上班，没有小孩","小孩还没到那个阶段呢","我们家小孩都上小学啦","没有没有，没有小孩","我们家没有一岁的娃","我都五十多了，哪来一岁的小孩","我们家小朋友刚出生","还没准备生小人","没有这么大的孩子","我们家崽都上幼儿园啦","我们家没有这么大的小孩","娃还在肚子里呢","刚结婚，还没打算要小孩","我女儿都十几岁了","我还在读书呢，没有小孩","宝宝还没到一岁","我们家女儿刚参加高考","小朋友还没有呢","小孩还没有这么大","我刚毕业，哪来的小孩","我家女儿都上初中了","我是单身，没有小孩","我儿子都上大学了"],"口碑差_7a8Ax7": ["你们骗人吧","不要冒充银行骗人好吧","你说你是不是骗子吧","你干嘛咯你是谁咯骗子吧","你们想骗我钱吧","你骗人吧","你骗来骗去有意思吗","电话诈骗要坐牢","你不要诈骗了","你骗来钱敢用吗","你骗人钱有命花吗","呃我这手机上怎么显示你诈骗","不会是诈骗电话吧","你骗来骗去以后要坐牢","你个骗子想骗我没那么容易","你不要骗我","不知道行了好不用应该都是诈骗哪个贷款","忽悠人骗人","行了行了行了唉呦不要干这种骗人勾当","你都是骗人我那个叫","骗人的吧","你到处骗人有什么好处","现在骗子这么多我凭什么相信你啊","你们骗子公司还是流氓公司啊","你骗钱要下地狱","你别骗人了吧","不知道九五九五二开头都是骗子","你不会是骗子吧","你不要骗人","那诈骗你这是类似诈骗电话","靠谱吗","你不要到处骗人今天你没有好下场","你们不会是骗子吧","嗯上介绍我骗","你骗人不好","电信诈骗吧","你告诉我你骗我有什么好处","嗯反正那上写一是诈骗","你这是诈骗电话吧","那个骗子","你你说你要骗人"]}
    def hnsw_index_update(self, from_json, to_dir, to_file):
        # 如果当前hnsw模型存在，则先删除再更新
        if os.path.exists(os.path.join(to_dir,to_file)):
            os.remove(os.path.join(to_dir,to_file))
        # file_name = os.path.basename(to_file).split(".")[0]
        # {parent_node_name:hnsw_index}
        self.hnsw_index[to_file] = self.build_hnsw_index(from_json, os.path.join(to_dir,to_file))
        # {parent_node_name:{"id_intent_dict":id_intent_dict,"id_sentence_dict":id_sentence_dict}
        self.id_sentence_intent_map[to_file] = self.json_deserialization(os.path.join(to_dir,to_file+".pkl"))
        logging.info("更新HNSW召回模型:{},更新HNSW召回模型映射关系:{}".format(os.path.join(to_dir,to_file+".bin"),os.path.join(to_dir,to_file+".pkl")))

    # 从原数据加载并构建hnsw图向量且保存至文件
    def build_hnsw_index(self, from_json, to_file):
        id_counter = 0
        sentence_intents = []
        sentence_vectors = []
        id_intent_dict = {}
        id_sentence_dict = {}
        id_embedding_dict = {}
        index = hnswlib.Index(space = 'cosine', dim = 512)
        index.init_index(max_elements=60000, ef_construction=400, M=64)
        for label,sentence_list in from_json.items():
            for sentence in sentence_list:
                # 本地加载并编码
                embedding = self.model.encode(sentence,show_progress_bar=False)
                sentence_intents.append(id_counter)
                sentence_vectors.append(embedding)
                id_intent_dict[id_counter] = label
                id_sentence_dict[id_counter] = sentence
                id_embedding_dict[id_counter] = embedding
                id_counter += 1
        index.add_items(sentence_vectors, sentence_intents)
        id_sentence_intent_dict = {"id_intent_dict":id_intent_dict,"id_sentence_dict":id_sentence_dict,"id_embedding_dict":id_embedding_dict} #id_intent字典, id_sentence字典 ,id_embedding字典
        self.json_serialize(to_file+".pkl",id_sentence_intent_dict)
        index.save_index(to_file+".bin")
        return index

    # 通过父亲节点以及query从对应的hnsw模型中召回数据
    # @profile
    def search(self, parent_node,query, k):
        id_intent_dict = self.id_sentence_intent_map[parent_node]["id_intent_dict"]
        id_sentence_dict = self.id_sentence_intent_map[parent_node]["id_sentence_dict"]
        id_embedding_dict = self.id_sentence_intent_map[parent_node]["id_embedding_dict"]
        query_embedding = self.model.encode(query.strip(),show_progress_bar=False)
        query_word2vec = self.sentence2vector(query.strip())
        labels, distances = self.hnsw_index[parent_node].knn_query(query_embedding, k=k)
        query_list = [query] * k
        query_embedding_list = [query_embedding] * k
        query_word2vec_list = [query_word2vec] * k
        recall_intents_list = []
        recall_sentences_list = []
        recall_distances_list = []
        recall_embedding_list = []
        recall_word2vec_list = []
        for index in range(k):
            intent = id_intent_dict[labels[0][index]]
            sentence = id_sentence_dict[labels[0][index]]
            distance = distances[0][index]
            embedding = id_embedding_dict[labels[0][index]]
            word2vector = self.sentence2vector(sentence)
            
            recall_intents_list.append(intent)
            recall_sentences_list.append(sentence)
            recall_distances_list.append(distance)
            recall_embedding_list.append(embedding)
            recall_word2vec_list.append(word2vector)

        return pd.DataFrame({'query':query_list,'query_embedding':query_embedding_list,'query_word2vec':query_word2vec_list,'recall_sentences':recall_sentences_list, 'recall_intents':recall_intents_list, 'recall_distances':recall_distances_list, 'recall_embedding':recall_embedding_list,'recall_word2vec':recall_word2vec_list})

    # 通过父亲节点以及query从对应的hnsw模型中召回数据
    # @profile
    def recall(self, parent_node,query, k):
        logging.info("######{}######".format(query))
        id_intent_dict = self.id_sentence_intent_map[parent_node]["id_intent_dict"]
        id_sentence_dict = self.id_sentence_intent_map[parent_node]["id_sentence_dict"]
        id_embedding_dict = self.id_sentence_intent_map[parent_node]["id_embedding_dict"]
        # 本地加载并编码
        # query_embedding = self.model.encode(query.strip(),show_progress_bar=False)
        # torchserve
        query_embedding_tmp = requests.post("http://127.0.0.1:8080/explanations/sbert",data={'data':query.strip()})
        query_embedding = np.array(query_embedding_tmp.json())
        logging.info("######{}######".format(len(query_embedding)))
        query_word2vec = self.sentence2vector(query.strip())
        labels, distances = self.hnsw_index[parent_node].knn_query(query_embedding, k=k)
        query_list = [query] * k
        query_embedding_list = [query_embedding] * k
        query_word2vec_list = [query_word2vec] * k
        recall_intents_list = []
        recall_sentences_list = []
        recall_distances_list = []
        recall_embedding_list = []
        recall_word2vec_list = []
        for index in range(k):
            intent = id_intent_dict[labels[0][index]]
            sentence = id_sentence_dict[labels[0][index]]
            distance = distances[0][index]
            embedding = id_embedding_dict[labels[0][index]]
            word2vector = self.sentence2vector(sentence)

            recall_intents_list.append(intent)
            recall_sentences_list.append(sentence)
            recall_distances_list.append(distance)
            recall_embedding_list.append(embedding)
            recall_word2vec_list.append(word2vector)

        # # 将字典数组 展开成数组字典
        # t = [dict(zip(tuple(a.keys()),t)) for t in list(zip(*(a.values())))]
        return {'query':query_list,'query_embedding':query_embedding_list,'query_word2vec':query_word2vec_list,'recall_sentences':recall_sentences_list, 'recall_intents':recall_intents_list, 'recall_distances':recall_distances_list, 'recall_embedding':recall_embedding_list,'recall_word2vec':recall_word2vec_list}

    # 通过父亲节点以及query从对应的hnsw模型中召回数据
    # @profile
    def recall_torchserve(self, parent_node,query, k):
        id_intent_dict = self.id_sentence_intent_map[parent_node]["id_intent_dict"]
        id_sentence_dict = self.id_sentence_intent_map[parent_node]["id_sentence_dict"]
        id_embedding_dict = self.id_sentence_intent_map[parent_node]["id_embedding_dict"]
        # 本地加载并编码
        # query_embedding = self.model.encode(query.strip(),show_progress_bar=False)
        # torchserve
        query_embedding_tmp = requests.post("http://127.0.0.1:8080/explanations/sbert",data={'data':query.strip()})
        query_embedding = np.array(query_embedding_tmp.json())
        query_word2vec = self.sentence2vector(query.strip())
        labels, distances = self.hnsw_index[parent_node].knn_query(query_embedding, k=k)
        
        ResDictList = []
        for index in range(k):
            intent = id_intent_dict[labels[0][index]]
            sentence = id_sentence_dict[labels[0][index]]
            distance = distances[0][index]
            embedding = id_embedding_dict[labels[0][index]]
            word2vector = self.sentence2vector(sentence)
            dict_tmp = {'query':query,'query_embedding':query_embedding,'query_word2vec':query_word2vec,'recall_sentences':sentence, 'recall_intents':intent, 'recall_distances':distance, 'recall_embedding':embedding,'recall_word2vec':word2vector}
            logging.info("dict:{}".format(dict_tmp))
            ResDictList.append(dict_tmp)

        return ResDictList

    # 加载hnsw向量图检索引擎
    def load_hnsw_index(self, hnsw_model_path):
        index = hnswlib.Index(space='cosine', dim = 512)
        index.load_index(hnsw_model_path)
        return index

    # json字符串序列化
    def json_serialize(self,to_file,json_str):
        with open(to_file, 'wb+') as f:
            data = pickle.dumps(json_str)
            f.write(data)
            
    # json字符串反序列化
    def json_deserialization(self,from_file):
        with open(from_file, 'rb+') as f:
            data = pickle.loads(f.read())
            return data

    # 中文语句去除特殊字符并转换成向量
    # @profile
    def sentence2vector(self, sent):
        vec_arr = []
        sentence = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+","", sent)
        for word in list(self.jieba.cut(sentence)):
            if word not in self.word2vec.wv.vocab.keys():
                vec_arr.append(np.random.randn(1, 300)) # 生成1行300列的标准正态分布向量
            else:
                vec_arr.append(self.word2vec.wv.get_vector(word))
        return np.mean(np.array(vec_arr, dtype=object), axis=0).reshape(1, -1)

    #计算语句向量的相似度
    def cos_sim(self, nd_a, nd_b):
        nd_a = np.array(nd_a)
        nd_b = np.array(nd_b)
        return np.sum(nd_a * nd_b) / (np.sqrt(np.sum(nd_a**2)) * np.sqrt(np.sum(nd_b**2)))

if __name__ == '__main__':
    sbert_model = SentenceTransformer(args.retrival_sbert)
    from_dict = {
        "姓名_oRqvs2": [
            "姓汤喝汤的汤",
            "免贵姓汤喝汤的汤",
            "我叫汤建飞",
            "我姓汤喝汤的汤",
            "我是汤建飞"
        ],
        "没小孩-幼儿3-8岁_EAdQu1": [
            "我都没有对象，怎么会有小孩",
            "我儿子都二十多岁了",
            "我们家宝宝三岁了",
            "我们家宝宝刚出生",
            "还没小人呢",
            "小孩刚参加中考",
            "一岁的小朋友没有",
            "我刚上班，没有小孩",
            "小孩还没到那个阶段呢",
            "我们家小孩都上小学啦",
            "没有没有，没有小孩",
            "我们家没有一岁的娃",
            "我都五十多了，哪来一岁的小孩",
            "我们家小朋友刚出生",
            "还没准备生小人",
            "没有这么大的孩子",
            "我们家崽都上幼儿园啦",
            "我们家没有这么大的小孩",
            "娃还在肚子里呢",
            "刚结婚，还没打算要小孩",
            "我女儿都十几岁了",
            "我还在读书呢，没有小孩",
            "宝宝还没到一岁",
            "我们家女儿刚参加高考",
            "小朋友还没有呢",
            "小孩还没有这么大",
            "我刚毕业，哪来的小孩",
            "我家女儿都上初中了",
            "我是单身，没有小孩",
            "我儿子都上大学了"
        ],
        "口碑差_7a8Ax7": [
            "你们骗人吧",
            "不要冒充银行骗人好吧",
            "你说你是不是骗子吧",
            "你干嘛咯你是谁咯骗子吧",
            "你们想骗我钱吧",
            "你骗人吧",
            "你骗来骗去有意思吗",
            "电话诈骗要坐牢",
            "你不要诈骗了",
            "你骗来钱敢用吗",
            "你骗人钱有命花吗",
            "呃我这手机上怎么显示你诈骗",
            "不会是诈骗电话吧",
            "你骗来骗去以后要坐牢",
            "你个骗子想骗我没那么容易",
            "你不要骗我",
            "不知道行了好不用应该都是诈骗哪个贷款",
            "忽悠人骗人",
            "行了行了行了唉呦不要干这种骗人勾当",
            "你都是骗人我那个叫",
            "骗人的吧",
            "你到处骗人有什么好处",
            "现在骗子这么多我凭什么相信你啊",
            "你们骗子公司还是流氓公司啊",
            "你骗钱要下地狱",
            "你别骗人了吧",
            "不知道九五九五二开头都是骗子",
            "你不会是骗子吧",
            "你不要骗人",
            "那诈骗你这是类似诈骗电话",
            "靠谱吗",
            "你不要到处骗人今天你没有好下场",
            "你们不会是骗子吧",
            "嗯上介绍我骗",
            "你骗人不好",
            "电信诈骗吧",
            "你告诉我你骗我有什么好处",
            "嗯反正那上写一是诈骗",
            "你这是诈骗电话吧",
            "那个骗子",
            "你你说你要骗人"
        ]
    }
    index = HNSW_Recall(sbert_model)
    index.hnsw_index_update(from_dict,"/home/cjzhu3/cjzhu3_nlu/from_dict_test.bin")
    res = index.search('我姓汤喝汤的汤',2)
    print(res)
