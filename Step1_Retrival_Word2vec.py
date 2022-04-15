import logging
import multiprocessing
from time import time
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser,Phrases
from Preprocessor import clean, read_file,read_file_zhongan
import args

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

"""
    1、Step1_Retrival_Word2vec:
        本脚本采用众安小贷业务AB意向会话语料中用户答话文本作为word2vec向量模型训练数据

    2、众安小贷语料提取脚本:
        hive -S -e "
        set mapreduce.job.queuename=aegis;
        select distinct aa.call_uuid,aa.callee,aa.dialog_text_path,aa.tag_name
        from (
        select task_id,call_uuid,callee,dialog_text_path,tag_name
        from dsj_ods.ods_call_info_hi 
        where tag_name in('A','B') and to_date(create_time) >= date_sub(current_date(),90)
        )aa inner join (
        select id 
        from dsj_ods.ods_call_task_hf 
        where task_name regexp '.*(众安小贷).*'
        )bb where aa.task_id=bb.id;">/home/cjzhu3/ZhongAnXiaoDai_Corpus.txt &
"""

"""
select dialog_text_path
from dsj_ods.ods_call_info_hi 
where tag_name in('A','B') limit 2;

select dialog_text_path
from dsj_ods.ods_call_info_hi 
where tag_name in('A','B') limit 2;

select split(dialog_text_path, '\n')
from dsj_ods.ods_call_info_hi 
where tag_name in('A','B') limit 2;

set mapreduce.job.queuename=aegis;
select count(distinct(dialog_text_path))
from dsj_ods.ods_call_info_hi 
where tag_name in('A','B');
"""

# 读取会话数据
def read_data(file_path):
    train = pd.DataFrame(read_file(file_path, True),columns=['session_id', 'role', 'content'])
    train['clean_content'] = train['content'].apply(clean)
    return train

# 读取众安对话数据中human数据
def read_data_zhongan(file_path):
    train = pd.DataFrame(read_file_zhongan(file_path,True),columns=["content"])
    train['clean_content'] = train['content'].apply(clean)
    return train

# 训练word2vec词向量模型    
def train_w2v(train, to_file):
    sent = [row.split() for row in train['clean_content']]
    phrases = Phrases(sent, min_count=5, progress_per=10000)
    bigram = Phraser(phrases)
    sentences = bigram[sent]

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count=2, window=2, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=15, workers=cores - 1, iter=7)

    t = time()
    w2v_model.build_vocab(sentences)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=15, report_delay=1)
    print('Time to train vocab: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.save(to_file)

if __name__ == "__main__":
    # train = read_data(args.train_raw_corpus)
    # train_w2v(train, args.retrival_word2vec)
    train = read_data_zhongan(args.train_raw_corpus_zhongan)
    train_w2v(train, args.retrival_word2vec_zhongan)