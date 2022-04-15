import sys
import os
import logging
import lightgbm as lgb
import pandas as pd
import joblib
from tqdm import tqdm
from Sort_Similarity_Calculater import Manual_Similarity
from sklearn.model_selection import train_test_split
# from Step2_Retrival_HNSW_FAISS import wam
import numpy as np
# from Bert_Similarity_Calculater import BertSimilarCalculate
import args

tqdm.pandas()

params = {'boosting_type': 'gbdt',
          'max_depth': 5,
          'objective': 'binary',
          'nthread': 3,  
          'num_leaves': 64,
          'learning_rate': 0.05,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 0.5,
          'subsample_freq': 5,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'max_position': 20,
          'group': 'name:groupId',
          'metric': 'auc'}

class Sort_LightGBM_Trainer(object):
    def __init__(self, model_path,gbm_train_path=args.sort_train_data,gbm_test_path=args.sort_test_data, do_train = True):
        self.manual_Sim = Manual_Similarity()
        # self.bert_Sim = BertSimilarCalculate()
        if do_train:
            logging.info('Training mode')
            self.train = pd.read_csv(gbm_train_path,delimiter="\t", encoding="utf-8")
            self.data = self.generate_feature(self.train)
            self.columns = [i for i in self.train.columns if 'question' not in i]
            self.trainer()
            self.save(model_path)
        else:
            logging.info('Predicting mode')
            self.test = pd.read_csv(gbm_test_path,delimiter="\t", encoding="utf-8") 
            self.gbm = joblib.load(model_path)

    def generate_feature(self, data):
        logging.info('Generating manual features.')
        data = pd.concat([data, pd.DataFrame.from_records(data.apply(lambda row: self.manual_Sim.Similarity_Calculate_ZhongAn(row['question1'] , row['question2']), axis=1))], axis=1)
        logging.info('Generating Bert features.')
        # data['bert_score'] = data.apply(lambda row: self.bert_Sim.predict(row['question1'] , row['question2'])[1], axis=1)
        return data

    def trainer(self):
        logging.info('Training lightgbm model.')
        self.gbm = lgb.LGBMRanker(**params)
        columns = [i for i in self.data.columns if i not in ['question1', 'intent1', 'question2','intent2', 'target']]
        X_train , X_test , y_train , y_test = train_test_split(self.data[columns] , self.data['target'] , test_size = 0.3 , random_state = 42)
        query_train = [X_train.shape[0]]
        query_val = [X_test.shape[0]]
        self.gbm.fit(X_train , y_train , group=query_train , eval_set=[(X_test , y_test)] , eval_group=[query_val] , eval_at=[5 , 10 , 20] , early_stopping_rounds=50)
    
    def save(self, model_path):
        logging.info('Saving lightgbm model.')
        joblib.dump(self.gbm, model_path)

    def predict(self , data: pd.DataFrame):
        columns = [i for i in data.columns if i not in ['question1', 'intent1', 'question2','intent2', 'target']]
        result = self.gbm.predict(data[columns])
        return result

if __name__ == '__main__':
    sort = Sort_LightGBM_Trainer(model_path=args.sort_lightgbm_model,gbm_train_path=args.sort_train_data_zhongan_cartesian,gbm_test_path=args.sort_test_data_zhongan_cartesian,do_train=True)