import logging
from collections import defaultdict
import jieba
from gensim import corpora, models
import args
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

"""
    1、Step3_Sort_Similarity_Trainer:
        本脚本基于训练样本数据(3486条)训练word2vec、fasttext、tfidf等模型，用于精排序阶段语句相似度计算以及有监督学习阶段的计算
"""

class Trainer(object):
    def __init__(self,stopwords_data = args.stopwords_data,sort_train_data=args.sort_train_data,sort_dev_data=args.sort_dev_data,sort_test_data=args.sort_test_data):
        self.data = self.data_reader(sort_train_data) + self.data_reader(sort_dev_data) + self.data_reader(sort_test_data)
        self.stopwords = open(stopwords_data , encoding='utf-8').readlines()
        self.preprocessor()
        self.train()
        self.saver()
    
    def data_reader(self , path):
        samples = []
        with open(path , 'r' , encoding='UTF-8') as f:
            for line in f:
                try:
                    sentence1, sentence2, target = line.split('\t')
                except Exception:
                    print('exception: ' , line)
                samples.append(sentence1)
                samples.append(sentence2)
        return samples
    
    def preprocessor(self):
        logging.info(" loading data.... ")
        self.data = [[word for word in jieba.cut(sentence) if word not in self.stopwords] for sentence in self.data]
        self.freq = defaultdict(int)
        for sentence in self.data:
            for word in sentence:
                self.freq[word] += 1
        self.data = [[word for word in sentence if self.freq[word] > 1] for sentence in self.data]
        logging.info(' building dictionary....')
        self.dictionary = corpora.Dictionary(self.data)
        self.dictionary.save(args.sort_dictionary)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        corpora.MmCorpus.serialize(args.sort_mmcorpus,self.corpus)

    def train(self):
        logging.info(' train tfidf model ...')
        self.tfidf = models.TfidfModel(self.corpus, normalize=True)
        logging.info(' train word2vec model...')
        self.w2v = models.Word2Vec(min_count=2,window=2,size=300,sample=6e-5,alpha=0.03,min_alpha=0.0007,negative=15,workers=8,iter=7)
        self.w2v.build_vocab(self.data)
        self.w2v.train(self.data,total_examples=self.w2v.corpus_count,epochs=15,report_delay=1)
        logging.info(' train fasttext model ...')
        self.fast = models.FastText(self.data,size=300,window=3,min_count=1,iter=10,min_n=3,max_n=6,word_ngrams=2)

    def saver(self):
        logging.info(' save tfidf model ...')
        self.tfidf.save(args.sort_tfidf_model)
        logging.info(' save word2vec model ...')
        self.w2v.save(args.sort_word2vec_model)
        logging.info(' save fasttext model ...')
        self.fast.save(args.sort_fasttext_model)

class Trainer_For_ZhongAn(object):
    def __init__(self):
        self.data = self.data_reader(args.sort_train_data_zhongan)
        self.stopwords = open(args.stopwords_data , encoding='utf-8').readlines()
        self.preprocessor()
        self.train()
        self.saver()
    
    def data_reader(self , path):
        samples = []
        with open(path , 'r' , encoding='UTF-8') as f:
            for line in f:
                try:
                    custom, intent = line.strip().split('\t')
                except Exception:
                    print('exception: ' , line)
                samples.append(custom)
        return samples
    
    def preprocessor(self):
        logging.info(" loading data.... ")
        self.data = [[word for word in jieba.cut(sentence) if word not in self.stopwords] for sentence in self.data]
        self.freq = defaultdict(int)
        for sentence in self.data:
            for word in sentence:
                self.freq[word] += 1
        self.data = [[word for word in sentence if self.freq[word] > 1] for sentence in self.data]
        logging.info(' building dictionary....')
        self.dictionary = corpora.Dictionary(self.data)
        self.dictionary.save(args.sort_dictionary_zhongan)
        self.corpus = [self.dictionary.doc2bow(text) for text in self.data]
        corpora.MmCorpus.serialize(args.sort_mmcorpus_zhongan,self.corpus)

    def train(self):
        logging.info(' train tfidf model ...')
        self.tfidf = models.TfidfModel(self.corpus, normalize=True)
        logging.info(' train word2vec model...')
        self.w2v = models.Word2Vec(min_count=2,window=2,size=300,sample=6e-5,alpha=0.03,min_alpha=0.0007,negative=15,workers=8,iter=7)
        self.w2v.build_vocab(self.data)
        self.w2v.train(self.data,total_examples=self.w2v.corpus_count,epochs=15,report_delay=1)
        logging.info(' train fasttext model ...')
        self.fast = models.FastText(self.data,size=300,window=3,min_count=1,iter=10,min_n=3,max_n=6,word_ngrams=2)

    def saver(self):
        logging.info(' save tfidf model ...')
        self.tfidf.save(args.sort_tfidf_model_zhongan)
        logging.info(' save word2vec model ...')
        self.w2v.save(args.sort_word2vec_model_zhongan)
        logging.info(' save fasttext model ...')
        self.fast.save(args.sort_fasttext_model_zhongan)
 
if __name__ == "__main__":
    # Trainer()
    Trainer_For_ZhongAn()