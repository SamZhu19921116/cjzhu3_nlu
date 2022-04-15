import sys
import pandas as pd
from sklearn.model_selection import train_test_split

inpath = sys.argv[1]

df_raw_0 = pd.read_csv(inpath,sep="\t",header=None,names=["custom","intent"])
df_raw_1 = df_raw_0[['custom','intent']].rename(columns={'custom': 'question1','intent':'intent1'})
df_raw_2 = df_raw_0[['custom','intent']].rename(columns={'custom': 'question2','intent':'intent2'})
df_raw_1['value'] = 1
df_raw_2['value'] = 1
df_raw_3 = df_raw_1.merge(df_raw_2,how='left',on='value')
df_raw_4 = df_raw_3.loc[(df_raw_3['question1'] != df_raw_3['question2'])]
df_raw_4[['question1','intent1','question2','intent2']].to_csv(inpath.replace(".csv","_Cartesian.csv"),index=None,header=None,sep="\t")
