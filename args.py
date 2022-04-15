import torch
import os

# root_path = os.path.abspath(os.path.dirname(r" "))
root_path = os.path.dirname(__file__)
# 原始对话语料
train_raw_corpus = os.path.join(root_path, 'data/chat.txt')
train_raw_corpus_zhongan = os.path.join(root_path, 'data/ZhongAnXiaoDai_Corpus_query.txt')
# 用于召回Retrival的word2vec Embedding
retrival_word2vec = os.path.join(root_path, "model/retrieval/word2vec")
retrival_word2vec_zhongan = os.path.join(root_path, "model/retrieval/word2vec_zhongan")
# Sentence-Bert官方预训练模型:该模型用于粗召回以及文本相似度度量
retrival_sbert = os.path.join(root_path, "sbert/model/distiluse-base-multilingual-cased-v1")
# HNSW parameters
retrival_hnsw_ef = 3000  # ef_construction defines a construction time/accuracy trade-off
retrival_hnsw_max = 64  # M defines tha maximum number of outgoing connections in the graph
retrival_hnsw_model = os.path.join(root_path, 'model/retrieval/hnsw.bin')
retrival_sbert_hnsw_model = os.path.join(root_path, 'model/retrieval/hnsw_sbert_zhongan.bin')
retrival_sbert_hnsw_model_credit_dir = os.path.join(root_path, 'model/retrieval/')
retrival_faiss_model = os.path.join(root_path, 'model/retrieval/faiss.bin')
retrival_faiss_word2vec_zhongan = os.path.join(root_path, 'model/retrieval/faiss_word2vec_zhongan.bin')
retrival_hnsw_train = os.path.join(root_path, 'data/retrieval/train.csv')
retrival_hnsw_train_zhongan = os.path.join(root_path, 'data/retrieval/train_zhongan.csv')
retrival_hnsw_credit_dir = os.path.join(root_path, 'data/retrieval/')
retrival_hnsw_dev = os.path.join(root_path, 'data/retrieval/dev.csv')
retrival_hnsw_test = os.path.join(root_path, 'data/retrieval/test.csv')
# sort阶段数据集
sort_train_data = os.path.join(root_path , 'data/sort/train.tsv')
sort_train_data_zhongan_cartesian = os.path.join(root_path , 'data/sort/train_zhongan_Cartesian_Target_Train.csv')
sort_train_data_zhongan = os.path.join(root_path, 'data/retrieval/train_zhongan.csv')
sort_dev_data = os.path.join(root_path , 'data/sort/dev.tsv')
sort_test_data = os.path.join(root_path , 'data/sort/test.tsv')
sort_test_data_zhongan_cartesian = os.path.join(root_path , 'data/sort/train_zhongan_Cartesian_Target_Train.csv')
sort_test_data_zhongan = os.path.join(root_path, 'data/retrieval/test_zhongan.csv')
# 停用词
stopwords_data = os.path.join(root_path, 'data/stopwords.txt') 
# sort语义相似度评估模型
sort_dictionary = os.path.join(root_path, 'model/sort/sort.dictionary')
sort_dictionary_zhongan = os.path.join(root_path, 'model/sort/sort.dictionary_zhongan')
sort_mmcorpus = os.path.join(root_path, 'model/sort/sort.mmcorpus')
sort_mmcorpus_zhongan = os.path.join(root_path, 'model/sort/sort.mmcorpus_zhongan')
# tfidf评估模型
sort_tfidf_model = os.path.join(root_path, 'model/sort/tfidf.model')
sort_tfidf_model_zhongan = os.path.join(root_path, 'model/sort/tfidf.model.zhongan')
# word2vec评估模型
sort_word2vec_model = os.path.join(root_path, 'model/sort/word2vec.model')
sort_word2vec_model_zhongan = os.path.join(root_path, 'model/sort/word2vec.model.zhongan')
# fasttext评估模型
sort_fasttext_model = os.path.join(root_path, 'model/sort/fasttext.model')
sort_fasttext_model_zhongan = os.path.join(root_path, 'model/sort/fasttext.model.zhongan')
# lightgbm模型对上述语义度量特征进行有监督训练并进行精排序
sort_lightgbm_model = os.path.join(root_path, 'model/sort/lightgbm')
sort_bm25_model = os.path.join(root_path, 'model/sort')
sep = '[SEP]'
# Bert
max_sequence_length = 103
chinese_bert_dir = os.path.join(root_path, 'lib/bert/')
chinese_bert_vocab = os.path.join(root_path, 'lib/bert/vocab.txt')
chinese_bert_config = os.path.join(root_path,'lib/bert/config.json')
similarity_bert_save = os.path.join(root_path, 'model/sort/best.pth.tar')
# 微调训练后的Sentence_Bert模型保存路径
similarity_sbert_save = os.path.join(root_path, 'sbert/model/fine-tuning')
max_length = 103
batch_size = 32
lr = 0.001
bert_chinese_model_path = os.path.join(root_path, 'lib/bert/pytorch_model.bin')
log_path = os.path.join(root_path, 'log/distil.log')
max_grad_norm = 5.0
gradient_accumulation = 2.0

# 是否采用GPU训练pytorch Bert模型
is_cuda = True
if is_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# dev_raw = os.path.join(root_path, 'data/开发集.txt')
# test_raw = os.path.join(root_path, 'data/测试集.txt')
# ware_path = os.path.join(root_path, 'data/ware.txt') 