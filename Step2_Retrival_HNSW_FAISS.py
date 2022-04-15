import logging
import os
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import args
from Preprocessor import clean
import hnswlib
import faiss

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

"""
    1、Step2_Retrival_HNSW_FAISS：
        本脚本提供两个用于粗召回阶段向量检索的工具：HNSW,FAISS
        其中输入为研究院已人工标注众安小贷数据：总共4981条，7/3分割训练样本3486条，测试样本1495条
"""

def wam(sentence, w2v_model):
    arr = []
    for s in clean(sentence).split():
        if s not in w2v_model.wv.vocab.keys():
            arr.append(np.random.randn(1, 300)) # 生成1行300列的标准正态分布向量
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr, dtype=object), axis=0).reshape(1, -1)
 
class HNSW_WORD2VEC(object):
    def __init__(self, w2v_path, data_path=None, ef=args.retrival_hnsw_ef, M=args.retrival_hnsw_max, hnsw_model_path=args.retrival_hnsw_model):
        self.w2v_model = KeyedVectors.load(w2v_path)

        self.data = self.data_load(data_path)
        if hnsw_model_path and os.path.exists(hnsw_model_path):
            self.hnsw = self.load_hnsw(hnsw_model_path) # 加载
        else:
            self.hnsw = self.build_hnsw(hnsw_model_path, ef=ef, m=M) # 训练

    def data_load(self, data_path):
        data = pd.read_csv(data_path,encoding='utf-8',header=None,names=["custom","intent"])
        data['custom_vec'] = data['custom'].apply(lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data

    def build_hnsw(self, to_file, ef=2000, m=64):
        logging.info('build_hnsw')
        dim = self.w2v_model.vector_size
        num_elements = self.data['custom'].shape[0]
        hnsw = np.stack(self.data['custom_vec'].values).reshape(-1, 300)
        p = hnswlib.Index(space='l2',dim=dim)  # possible options are l2, cosine or ip
        p.init_index(max_elements=num_elements, ef_construction=ef, M=m)
        p.set_ef(10)
        p.set_num_threads(8)
        p.add_items(hnsw)
        logging.info('Start')
        labels, distances = p.knn_query(hnsw, k=1)
        print('labels: ', labels)
        print('distances: ', distances)
        logging.info("Recall:{}".format(np.mean(labels.reshape(-1) == np.arange(len(hnsw)))))
        p.save_index(to_file)
        return p

    def load_hnsw(self, hnsw_model_path):
        hnsw = hnswlib.Index(space='l2', dim=self.w2v_model.vector_size)
        hnsw.load_index(hnsw_model_path)
        return hnsw

    def search(self, text, k=5):
        test_vec = wam(clean(text), self.w2v_model)
        q_labels, q_distances = self.hnsw.knn_query(test_vec, k=k)
        return pd.concat((self.data.iloc[q_labels[0]]['custom'].reset_index(),self.data.iloc[q_labels[0]]['assistance'].reset_index(drop=True),pd.DataFrame(q_distances.reshape(-1, 1), columns=['q_distance'])),axis=1)

from sentence_transformers import SentenceTransformer
class HNSW_SBERT(object):
    def __init__(self, sbert_model, data_path=None, ef=args.retrival_hnsw_ef, M=args.retrival_hnsw_max, hnsw_sbert_model_path=args.retrival_sbert_hnsw_model):
        if isinstance(sbert_model,SentenceTransformer):
            self.model = sbert_model
        elif isinstance(sbert_model,str):
            self.model = SentenceTransformer(sbert_model)
        else:
            raise TypeError('Input Type Error !')

        self.data = self.data_load_zhongan(data_path)
        if hnsw_sbert_model_path and os.path.exists(hnsw_sbert_model_path):
            self.hnsw = self.load_hnsw(hnsw_sbert_model_path) # 加载
        else:
            self.hnsw = self.build_hnsw(hnsw_sbert_model_path, ef=ef, m=M) # 训练

    def data_load_zhongan(self, data_path):
        data = pd.read_csv(data_path,sep = "\t",names = ['custom','intent'])
        data['custom_vec'] = data['custom'].apply(lambda x: self.model.encode(x.strip()))
        data = data.dropna()
        return data

    def build_hnsw(self, to_file, ef=400, m=64):
        logging.info('build_hnsw')
        index = hnswlib.Index(space = 'cosine', dim = 512)
        index.init_index(max_elements = len(self.data['custom_vec'].values), ef_construction = ef, M = m)
        index.set_ef(10)
        index.set_num_threads(8)
        index.add_items(np.stack(self.data['custom_vec'].values).reshape(-1, 512))
        print("Saving index to:", to_file)
        index.save_index(to_file)
        return index

    def load_hnsw(self, hnsw_model_path):
        hnsw = hnswlib.Index(space='cosine', dim = 512)
        hnsw.load_index(hnsw_model_path)
        return hnsw

    def search(self, text, k=5):
        text_vec = self.model.encode(text.strip())
        q_labels, q_distances = self.hnsw.knn_query(text_vec, k=k)
        return pd.concat((self.data.iloc[q_labels[0]]['custom'].reset_index(),self.data.iloc[q_labels[0]]['intent'].reset_index(drop=True),pd.DataFrame(q_distances.reshape(-1, 1), columns=['q_distance'])),axis=1)

class FAISS_WORD2VEC(object):
    def __init__(self, w2v_path, data_path=None,faiss_model_path=args.retrival_faiss_model):
        self.w2v_model = KeyedVectors.load(w2v_path)

        self.data = self.data_load_zhongan(data_path)
        if faiss_model_path and os.path.exists(faiss_model_path):
            self.faiss = self.load_faiss(faiss_model_path) # 加载
        else:
            self.faiss = self.build_faiss(faiss_model_path) # 训练
            #self.faiss = self.build_faiss_hnsw(faiss_model_path)

    def data_load(self, data_path):
        data = pd.read_csv(data_path)
        data['custom_vec'] = data['custom'].apply(lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data

    def data_load_zhongan(self, data_path):
        data = pd.read_csv(data_path,sep = "\t",names = ['custom','intent'])
        data['custom_vec'] = data['custom'].apply(lambda x: wam(x, self.w2v_model))
        data['custom_vec'] = data['custom_vec'].apply(lambda x: x[0][0] if x.shape[1] != 300 else x)
        data = data.dropna()
        return data

    # The most basic nearest neighbor search by L2 distance. This is much faster than scipy. You should first try this, especially when the database is relatively small (N<10^6). The search is automatically paralellized.
    def build_faiss(self, to_file):
        logging.info('build_faiss')
        dim = self.w2v_model.vector_size
        vec_data = np.stack(self.data['custom_vec'].values).reshape(-1, 300).astype(np.float32)
        index = faiss.IndexFlatL2(dim)
        index.add(vec_data)
        distances , labels = index.search(vec_data, k=1)
        print('labels:', labels)
        print('distances:', distances)
        logging.info("Recall:{}".format(np.mean(labels.reshape(-1) == np.arange(len(vec_data)))))
        faiss.write_index(index, to_file)
        return index
    
    # There are several methods for ANN search. Currently, HNSW + IVFPQ achieves the best performance for billion-scale data in terms of the balance among memory, accuracy, and runtime.
    def build_faiss_hnsw(self, to_file,M=16,NBits=8,NList=1000,HNSW_M=32,NProbe=8):
        # M = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
        # nbits = 8 # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
        # Param of IVF
        # nlist = 1000  # The number of cells (space partition). Typical value is sqrt(N)
        # Param of HNSW
        # hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32
        logging.info('build_faiss_hnsw')
        dim = self.w2v_model.vector_size
        vec_data = np.stack(self.data['custom_vec'].values).reshape(-1, 300).astype(np.float32)
        # Setup
        quantizer = faiss.IndexHNSWFlat(dim, HNSW_M)
        index = faiss.IndexIVFPQ(quantizer, dim, NList, M, NBits)
        # Train
        index.train(vec_data)
        print(index.is_trained)
        # Add
        index.add(vec_data)
        # Search
        index.nprobe = NProbe # Runtime param. The number of cells that are visited for search.
        distances , labels = index.search(vec_data, k=1)
        print('labels:', labels)
        print('distances:', distances)
        logging.info("Recall:{}".format(np.mean(labels.reshape(-1) == np.arange(len(vec_data)))))
        faiss.write_index(index, to_file)
        return index

    def load_faiss(self, faiss_model_path):
        faiss_model = faiss.read_index(faiss_model_path)
        return faiss_model

    def search(self, text, k=5):
        test_vec = wam(clean(text), self.w2v_model)
        test_vec = test_vec.astype(np.float32)
        q_distances,q_labels = self.faiss.search(test_vec, k=k)
        return pd.concat((self.data.iloc[q_labels[0]]['custom'].reset_index(),self.data.iloc[q_labels[0]]['intent'].reset_index(drop=True),pd.DataFrame(q_distances.reshape(-1, 1), columns=['q_distance'])),axis=1)

if __name__ == "__main__":
    test = '我不是本人你打错了'
    node_order = {1:"Credit_BWork",2:"Credit_OpeningRemarks2",3:"Credit_Surname",4:"Credit_Education",5:"Credit_Invite",6:"Credit_BusinessHours",7:"Credit_OffHook"}
    for index,name in node_order.items():
        hnsw = HNSW_SBERT(args.retrival_sbert,os.path.join(args.retrival_hnsw_credit_dir,"{}.csv".format(name)),args.retrival_hnsw_ef,args.retrival_hnsw_max,os.path.join(args.retrival_sbert_hnsw_model_credit_dir,"{}.bin".format(name)))
        hnsw_result = hnsw.search(test, k=5)
        print(hnsw_result)

    # #######################################################################################################
    # test = '我不是本人你打错了'
    # # hnsw = HNSW_WORD2VEC(args.retrival_word2vec_zhongan,args.retrival_hnsw_train_zhongan,args.retrival_hnsw_ef,args.retrival_hnsw_max,args.retrival_word2vec_hnsw_model)
    # # hnsw_result = hnsw.search(test, k=10)
    # # print(hnsw_result)

    # #######################################################################################################
    # # hnsw = HNSW_SBERT(args.similarity_sbert_save,args.retrival_hnsw_train_zhongan,args.retrival_hnsw_ef,args.retrival_hnsw_max,args.retrival_sbert_hnsw_model)
    # # hnsw_result = hnsw.search(test, k=1)
    # # print(hnsw_result)

    # # hnsw = HNSW_SBERT(args.retrival_sbert,args.retrival_hnsw_train_zhongan,args.retrival_hnsw_ef,args.retrival_hnsw_max,args.retrival_sbert_hnsw_model)

    # # # 召回topN问题及精排序
    # # def Rough_Recall(query,intent_true,topN = 10):
    # #     #print("query:{0},intent_true:{1}".format(query,intent_true))
    # #     temp = pd.DataFrame({'query': [query]*topN, 'intent_true':[intent_true]*topN, 'retrieved': hnsw.search(query, topN)['custom'], 'intent_predict': hnsw.search(query, topN)['intent']})
    # #     return temp

    # # import pandas as pd
    # # df_test = pd.read_csv(args.sort_test_data_zhongan,sep="\t",header=None,names=["custom","intent_true"])
    # # DF_Rough_Recall = pd.concat([Rough_Recall(row['custom'],row['intent_true']) for index, row in df_test.iterrows()], ignore_index=True)
    # # DF_Rough_Recall.to_csv("result/rough_recall.csv", mode='w', index=False)

    # # DF_Rough_Recall_Same = DF_Rough_Recall.loc[(DF_Rough_Recall['intent_true'] == DF_Rough_Recall['intent_predict'])]
    # # rough_recall = len(DF_Rough_Recall_Same) / len(DF_Rough_Recall)
    # # print("recall:{}".format(rough_recall))
    # #######################################################################################################
    # faiss = FAISS_WORD2VEC(w2v_path=args.retrival_word2vec_zhongan, data_path=args.retrival_hnsw_train_zhongan,faiss_model_path=args.retrival_faiss_word2vec_zhongan)
    # # faiss_result = faiss.search(test, k=10)
    # # print(faiss_result)

    # # 召回topN问题及精排序
    # def Rough_Recall(query,intent_true,topN = 10):
    #     #print("query:{0},intent_true:{1}".format(query,intent_true))
    #     temp = pd.DataFrame({'query': [query]*topN, 'intent_true':[intent_true]*topN, 'retrieved': faiss.search(query, topN)['custom'], 'intent_predict': faiss.search(query, topN)['intent']})
    #     return temp

    # import pandas as pd
    # df_test = pd.read_csv(args.sort_test_data_zhongan,sep="\t",header=None,names=["custom","intent_true"])
    # DF_Rough_Recall = pd.concat([Rough_Recall(row['custom'],row['intent_true']) for index, row in df_test.iterrows()], ignore_index=True)
    # DF_Rough_Recall.to_csv("result/rough_recall.csv", mode='w', index=False)

    # DF_Rough_Recall_Same = DF_Rough_Recall.loc[(DF_Rough_Recall['intent_true'] == DF_Rough_Recall['intent_predict'])]
    # rough_recall = len(DF_Rough_Recall_Same) / len(DF_Rough_Recall)
    # print("recall:{}".format(rough_recall))