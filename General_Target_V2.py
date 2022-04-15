import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def Auto_Similar_Calculate(intent_A,intent_B):
    if intent_A.strip() == intent_B.strip():
        return {"target":"1"}
    else:
        return {"target":"0"}

inpath = sys.argv[1]

df_raw_0 = pd.read_csv(inpath,sep="\t",header=None,names=['question1','intent1','question2','intent2'])
df_raw_0['target'] = pd.DataFrame.from_records(df_raw_0.apply(lambda row:Auto_Similar_Calculate(row['intent1'],row['intent2']), axis=1))

df_raw_1_same = df_raw_0.loc[(df_raw_0['question1'] != df_raw_0['question2']) & (df_raw_0['intent1'] == df_raw_0['intent2'])]

df_raw_1_same_len = len(df_raw_1_same)

df_raw_1_not_same = df_raw_0.loc[(df_raw_0['question1'] != df_raw_0['question2']) & (df_raw_0['intent1'] != df_raw_0['intent2'])]

df_raw_1_not_same_len = len(df_raw_1_not_same)

ratio = (df_raw_1_same_len * 4) / df_raw_1_not_same_len

train_not_same_sample, test_1 = train_test_split(df_raw_1_not_same[['question1','intent1','question2','intent2','target']], train_size=ratio, random_state=0)

print(len(train_not_same_sample))
#total = pd.concat(df_raw_1_same,train_not_same_sample,axis=0)
total = df_raw_1_same.append(train_not_same_sample)
total[['question1','intent1','question2','intent2','target']].to_csv(inpath.replace('.csv','_Ratio.csv'),index=None,sep="\t",header=True)
