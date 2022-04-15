from copy import deepcopy
from random import randint

def shuffle(lst):
  temp_lst = deepcopy(lst)
  m = len(temp_lst)
  while (m):
    m -= 1
    i = randint(0, m)
    temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
  return temp_lst

import pandas as pd
same_question = pd.read_csv(r"/home/cjzhu3/NLU_CJZHU3/SBert/data/train_zhongan_Cartesian_Same_Train.csv",sep="\t",header=None,names=["question1","intent1","question2","intent2"])
Ko_list = same_question.question1.to_list()
Cn_list = same_question.question2.to_list()
shuffle_Cn_list = shuffle(Cn_list)
shuffle_Ko_list = shuffle(Ko_list)

from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader

train_size = int(len(Ko_list) * 0.8)
eval_size = len(Ko_list) - train_size

# Define your train examples.
train_data = []
for idx in range(train_size):
  train_data.append(InputExample(texts=[Ko_list[idx], Cn_list[idx]], label=1.0))
  train_data.append(InputExample(texts=[shuffle_Ko_list[idx], shuffle_Cn_list[idx]], label=0.0))

# Define your evaluation examples
sentences1 = Ko_list[train_size:]
sentences2 = Cn_list[train_size:]
sentences1.extend(list(shuffle_Ko_list[train_size:]))
sentences2.extend(list(shuffle_Cn_list[train_size:]))
scores = [1.0] * eval_size + [0.0] * eval_size

#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('/home/cjzhu3/NLU_CJZHU3/SBert/distiluse-base-multilingual-cased-v1')
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
# Define your train dataset, the dataloader and the train loss
train_dataset = SentencesDataset(train_data, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100, evaluator=evaluator, evaluation_steps=100, output_path='/home/cjzhu3/NLU_CJZHU3/SBert/model')
