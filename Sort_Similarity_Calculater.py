import logging
import jieba
from jieba import posseg
import numpy as np
import difflib
from gensim import corpora, models
# from Step2_Retrival_HNSW_FAISS import wam
import args
import re
import os

#os.remove("/tmp/jieba.cache") #每次删除结巴缓存
#jieba.load_userdict("/usr/local/anaconda3/envs/nlu/lib/python3.7/site-packages/jieba/dict.txt") # 用结巴全局加载 保证分析效率

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def clean(sent, sep='<'):
    sent = re.sub(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+","", sent)
    i = 0
    tmp = []
    while i < len(sent):
        if sent[i] != sep:
            tmp.append(sent[i])
            i += 1
        else:
            tmp.append(sent[i:i + 5])
            i += 5
    # 过滤短文本？
    return " ".join(tmp)

def wam(sentence, w2v_model):
    arr = []
    for s in clean(sentence).split():
        if s not in w2v_model.wv.vocab.keys():
            arr.append(np.random.randn(1, 300)) # 生成1行300列的标准正态分布向量
        else:
            arr.append(w2v_model.wv.get_vector(s))
    return np.mean(np.array(arr, dtype=object), axis=0).reshape(1, -1)

class Similarity_Measure(object):
    def __init__(self):
        self.dictionary = corpora.Dictionary.load(args.sort_dictionary_zhongan)
        self.corpus = corpora.MmCorpus(args.sort_mmcorpus_zhongan)
        self.tfidf = models.TfidfModel.load(args.sort_tfidf_model_zhongan)
        self.w2v_model = models.KeyedVectors.load(args.sort_word2vec_model_zhongan)
        self.fasttext = models.FastText.load(args.sort_fasttext_model_zhongan)

    # dif字符串相似度
    def Difflib_SeqMatcher(self , str_a , str_b):
        return difflib.SequenceMatcher(None, str_a, str_b).ratio()
    
    # 最长公共子序列文本相似度
    def Lcs_Score(self , str_a , str_b):
        lengths = [[0 for j in range(len(str_b) + 1 )] for i in range(len(str_a) + 1)]
        for i,x in enumerate(str_a):
            for j,y in enumerate(str_b):
                if x==y:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j] , lengths[i][j+1])

        result = ""
        x,y = len(str_a) , len(str_b)
        while x !=0 and y !=0:
            if lengths[x][y] == lengths[x - 1][y]:
                x -= 1
            elif lengths[x][y] == lengths[x][y-1]:
                y -= 1
            else:
                assert str_a[x-1] == str_b[y-1]
                result = str_a[x-1] + result
                x -= 1
                y -= 1
        
        longestdist = lengths[len(str_a)][len(str_b)]
        ratio = longestdist / min(len(str_a) , len(str_b))
        return ratio

    # Levenshtein编辑距离
    def Levenshtein_Distance(self , str1 , str2):
        m = len(str1)
        n = len(str2)
        lensum = float(m + n)
        d = [[0] * (n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        
        for j in range(1 , n+1):
            for i in range(1 , m+1):
                if str1[i -1] == str2[j -1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(d[i-1][j] , d[i][j-1] , d[i-1][j-1]) + 1
        dist = d[-1][-1]
        ratio = (lensum -dist) / lensum
        return ratio
    # @profile
    # @classmethod
    def tokenize(self , str_a):
        wordsa = posseg.cut(str_a) 
        cuta = ""
        seta = set()
        for key in wordsa:
            cuta += key.word + " "
            seta.add(key.word)
        return [cuta , seta]
    #@profile
    def tokenizer(self, str_a):
        return " ".join(jieba.cut(str_a))

    # 字符串雅可比相似度
    def JaccardSim(self , str_a , str_b):
        seta = self.tokenize(str_a)[1]
        setb = self.tokenize(str_b)[1]
        sa_sb = 1.0 * len(seta & setb) / len(seta | setb)
        return sa_sb
    # 向量余弦相似度
    @staticmethod
    def cos_sim(a ,b):
        a = np.array(a)
        b = np.array(b)
        return np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
    # 向量欧式距离相似度
    @staticmethod
    def eucl_sim(a ,b):
        a = np.array(a)
        b = np.array(b)
        return 1 / (1 + np.sqrt((np.sum(a - b)**2)))
    # 向量皮尔逊相似度
    @staticmethod
    def pearson_sim(a , b):
        a = np.array(a)
        b = np.array(b)
        a = a - np.average(a)
        b = b - np.average(b)
        return np.sum(a * b) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
    # @profile
    def tokenSimilarity(self , str_a , str_b , method='w2v' , sim='cos'):
        # str_a = self.tokenize(str_a)[0]
        # str_b = self.tokenize(str_b)[0]
        str_a = self.tokenizer(str_a)
        str_b = self.tokenizer(str_b)
        vec_a , vec_b , model  = None , None , None
        if method == 'w2v':
            vec_a = wam(str_a , self.w2v_model)
            vec_b = wam(str_b , self.w2v_model)
            # model = self.w2v_model
        elif method == 'fasttext':
            vec_a = wam(str_a, self.fasttext)
            vec_b = wam(str_b, self.fasttext)
            # model = self.fasttext
        elif method == 'tfidf':
            vec_a = np.array(self.tfidf[self.dictionary.doc2bow(str_a.split())]).mean()
            vec_b = np.array(self.tfidf[self.dictionary.doc2bow(str_b.split())]).mean()
        else:
            NotImplementedError
        result = None

        if (vec_a is not None) and (vec_b is not None):
            if sim == 'cos':
                result = self.cos_sim(vec_a, vec_b)
            elif sim == 'eucl':
                result = self.eucl_sim(vec_a, vec_b)
            elif sim == 'pearson':
                result = self.pearson_sim(vec_a, vec_b)
            # elif sim == 'wmd' and model:
            #     result = model.wmdistance(str_a, str_b)
        return result

    # 相似度度量
    def Similarity_Calculate(self, str1, str2):
        return {
            'lcs_score':
            self.Lcs_Score(str1, str2),
            'edit_dist':
            self.Levenshtein_Distance(str1, str2),
            'diff_score':
            self.Difflib_SeqMatcher(str1,str2),
            'jaccard':
            self.JaccardSim(str1, str2),
            'w2v_cos':
            self.tokenSimilarity(str1, str2, method='w2v', sim='cos'),
            'w2v_eucl':
            self.tokenSimilarity(str1, str2, method='w2v', sim='eucl'),
            'w2v_pearson':
            self.tokenSimilarity(str1, str2, method='w2v', sim='pearson'),
            'w2v_wmd':
            self.tokenSimilarity(str1, str2, method='w2v', sim='wmd'),
            'fast_cos':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='cos'),
            'fast_eucl':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='eucl'),
            'fast_pearson':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='pearson'),
            'fast_wmd':
            self.tokenSimilarity(str1, str2, method='fasttest', sim='wmd'),
            'tfidf_cos':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='cos'),
            'tfidf_eucl':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='eucl'),
            'tfidf_pearson':
            self.tokenSimilarity(str1, str2, method='tfidf', sim='pearson')
        }
    
    def Similarity_Calculate_V1(self, str1, str2,str3):
        # {'query','retrieved','intent','lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson'}
        return {
            'query':
            str1,
            'retrieved':
            str2,
            'intent':
            str3,
            'lcs_score':
            self.Lcs_Score(str1, str2),
            'edit_dist':
            self.Levenshtein_Distance(str1, str2),
            'diff_score':
            self.Difflib_SeqMatcher(str1,str2),
            'jaccard':
            self.JaccardSim(str1, str2),
            'w2v_cos':
            self.tokenSimilarity(str1, str2, method='w2v', sim='cos'),
            'w2v_eucl':
            self.tokenSimilarity(str1, str2, method='w2v', sim='eucl'),
            'w2v_pearson':
            self.tokenSimilarity(str1, str2, method='w2v', sim='pearson'),
            # 'w2v_wmd':
            # self.tokenSimilarity(str1, str2, method='w2v', sim='wmd'),
            'fast_cos':
            self.tokenSimilarity(str1, str2, method='fasttext', sim='cos'),
            'fast_eucl':
            self.tokenSimilarity(str1, str2, method='fasttext', sim='eucl'),
            'fast_pearson':
            self.tokenSimilarity(str1, str2, method='fasttext', sim='pearson'),
            # 'fast_wmd':
            # self.tokenSimilarity(str1, str2, method='fasttext', sim='wmd'),
            # 'tfidf_cos':
            # self.tokenSimilarity(str1, str2, method='tfidf', sim='cos'),
            # 'tfidf_eucl':
            # self.tokenSimilarity(str1, str2, method='tfidf', sim='eucl'),
            # 'tfidf_pearson':
            # self.tokenSimilarity(str1, str2, method='tfidf', sim='pearson')
        }

    def Similarity_Calculate_ZhongAn(self, str1, str2):
        return {
            'lcs_score':
            self.Lcs_Score(str1, str2),
            'edit_dist':
            self.Levenshtein_Distance(str1, str2),
            'diff_score':
            self.Difflib_SeqMatcher(str1,str2),
            'jaccard':
            self.JaccardSim(str1, str2),
            'w2v_cos':
            self.tokenSimilarity(str1, str2, method='w2v', sim='cos'),
            'w2v_eucl':
            self.tokenSimilarity(str1, str2, method='w2v', sim='eucl'),
            'w2v_pearson':
            self.tokenSimilarity(str1, str2, method='w2v', sim='pearson'),
            # 'w2v_wmd':
            # self.tokenSimilarity(str1, str2, method='w2v', sim='wmd'),
            'fast_cos':
            self.tokenSimilarity(str1, str2, method='fasttext', sim='cos'),
            'fast_eucl':
            self.tokenSimilarity(str1, str2, method='fasttext', sim='eucl'),
            'fast_pearson':
            self.tokenSimilarity(str1, str2, method='fasttext', sim='pearson'),
            # 'fast_wmd':
            # self.tokenSimilarity(str1, str2, method='fasttext', sim='wmd'),
            # 'tfidf_cos':
            # self.tokenSimilarity(str1, str2, method='tfidf', sim='cos'),
            # 'tfidf_eucl':
            # self.tokenSimilarity(str1, str2, method='tfidf', sim='eucl'),
            # 'tfidf_pearson':
            # self.tokenSimilarity(str1, str2, method='tfidf', sim='pearson')
        }
    # @profile
    def Similarity_Calculate_V2(self, _query, _retrieved, _intent):
        return {
            'query': _query,
            'retrieved': _retrieved,
            'intent': _intent,
            'diff_score': self.Difflib_SeqMatcher(_query, _retrieved),
            'w2v_cos': self.tokenSimilarity(_query, _retrieved, method='w2v', sim='cos'),
        }

if __name__ == '__main__':
    sim = Similarity_Measure()

    while(True):
        sentence1 = input('sentence1: ')
        sentence2 = input('sentence2: ')
        sim_dict = sim.Similarity_Calculate_V2(sentence1 , sentence2,"投诉")
        print(sim_dict)