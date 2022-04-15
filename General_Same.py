import pandas as pd
import sys

inpath = sys.argv[1]

df_raw_0 = pd.read_csv(inpath,sep="\t",header=None,names=['question1','intent1','question2','intent2'])
df_raw_1 = df_raw_0.loc[(df_raw_0['question1'] != df_raw_0['question2']) & (df_raw_0['intent1'] == df_raw_0['intent2'])]
df_raw_1.to_csv(inpath.replace(".csv","_Same.csv"),sep="\t",header=None,index=None)
