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
print(df_raw_0)
train_0, test_0 = train_test_split(df_raw_0[['question1','intent1','question2','intent2','target']], test_size=0.05, random_state=0)
print(test_0)
train_1, test_1 = train_test_split(test_0[['question1','intent1','question2','intent2','target']], test_size=0.3, random_state=0)
print(test_1)
train_1[['question1','intent1','question2','intent2','target']].to_csv(inpath.replace('.csv','_Target_Train.csv'),index=None,sep="\t",header=True)
test_1[['question1','intent1','question2','intent2','target']].to_csv(inpath.replace('.csv','_Target_Test.csv'),index=None,sep="\t",header=True)
