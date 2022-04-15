# -*- coding: utf-8 -*-
import os
import args
import pandas as pd
import logging
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer(args.retrival_sbert)
# 基于hnsw的query问题的粗召回(按节点分层，加载每层节点的召回模型)
from Step2_Retrival_HNSW_FAISS import HNSW_SBERT
CNodeName_ENodeName_Dict = {"工作":"Credit_BWork","开场白":"Credit_OpeningRemarks2","姓氏":"Credit_Surname","学历":"Credit_Education","邀约":"Credit_Invite","营执时间":"Credit_BusinessHours","摘机":"Credit_OffHook"}
hnsw_node_dict = {}
for CNodeName,ENodeName in CNodeName_ENodeName_Dict.items():
    hnsw = HNSW_SBERT(sbert_model,os.path.join(args.retrival_hnsw_credit_dir,"{}.csv".format(ENodeName)),args.retrival_hnsw_ef,args.retrival_hnsw_max,os.path.join(args.retrival_sbert_hnsw_model_credit_dir,"{}.bin".format(ENodeName)))
    hnsw_node_dict[ENodeName] = hnsw
# 基于多种度量方式的计算两句子的相似度
from Sort_Similarity_Calculater import Similarity_Measure
sim_calc = Similarity_Measure()
# 双塔BERT文本相似度度量
from Sentence_Bert_Calculater import SBertSimCalc
sbert_sim = SBertSimCalc(sbert_model=sbert_model)

# 召回topN问题及精排序
def Fine_Sort_V1(query,topN = 10):
    #query问题粗召回
    RoughRecall = pd.DataFrame.from_records({'query': [query]*topN ,'retrieved': hnsw.search(query, topN)['custom'] , 'intent': hnsw.search(query, topN)['intent']}) 
    # 多种度量方式计算query问题以及query召回问题的相似度
    FineSort = pd.DataFrame.from_records(RoughRecall.apply(lambda row: sim_calc.Similarity_Calculate_ZhongAn_V2(row['query'], row['retrieved'], row['intent']), axis=1))
    FineSort['sbert_score'] = FineSort.apply(lambda row: sbert_sim.predict(row['query'], row['retrieved']) , axis=1)
    return FineSort

# 召回topN问题及精排序(按节点分层，每层分不同召回集)
def Fine_Sort_V2(node_name,query,topN = 10):
    #query问题粗召回
    RoughRecall = pd.DataFrame.from_records({'query': [query]*topN ,'retrieved': hnsw_node_dict[node_name].search(query, topN)['custom'] , 'intent': hnsw_node_dict[node_name].search(query, topN)['intent']}) 
    # 多种度量方式计算query问题以及query召回问题的相似度
    FineSort = pd.DataFrame.from_records(RoughRecall.apply(lambda row: sim_calc.Similarity_Calculate_ZhongAn_V2(row['query'], row['retrieved'], row['intent']), axis=1))
    FineSort['sbert_score'] = FineSort.apply(lambda row: sbert_sim.predict(row['query'], row['retrieved']) , axis=1)
    return FineSort

# 通过classifaction_report生成相应结果报告
def classifaction_report_to_csv(y_true, y_pred):
    from sklearn.metrics import classification_report,confusion_matrix
    # 通过混淆矩阵计算整体ACC
    con_mat = confusion_matrix(y_true,y_pred)
    cor_i = [con_mat[i][i] for i in range(len(con_mat))]
    total_acc = sum(cor_i) / sum(map(sum, con_mat))
    # 计算各分类指标
    report_data = []
    lines = classification_report(y_true, y_pred).split('\n')
    # 普通分类列
    for line in lines[2:]:
        row = {}
        data = line.split()
        if len(data) == 5:
            row['class'] = data[0]
            row['precision'] = data[1]
            row['recall'] = data[2]
            row['f1_score'] = data[3]
            row['support'] = data[4]
            report_data.append(row)
        elif len(data) == 6:
            row['class'] = data[0]+'_'+data[1]
            row['precision'] = data[2]
            row['recall'] = data[3]
            row['f1_score'] = data[4]
            row['support'] = data[5]
            report_data.append(row)
        elif len(data) == 3:
            row['class'] = data[0]
            row['precision'] = str(total_acc)+"(整体ACC)"
            row['recall'] = ""
            row['f1_score'] = data[1]
            row['support'] = data[2]
            report_data.append(row)
    return pd.DataFrame.from_dict(report_data)

# 通过正则算法过滤文本中的非法字符
import re
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
    return "".join(tmp)

# 关键词奖惩(通配权值) + 语义评估(sbert + word2vec)
# node_name = {"Credit_BWork","Credit_OpeningRemarks2","Credit_Surname","Credit_Education","Credit_Invite","Credit_BusinessHours","Credit_OffHook"}
# nodename_dict = {"Credit_BWork":Credit_BWork_name_rule_dict,"Credit_OpeningRemarks2":Credit_OpeningRemarks2_name_rule_dict,"Credit_Surname":Credit_Surname_name_rule_dict,"Credit_Education":Credit_Education_name_rule_dict,"Credit_Invite":Credit_Invite_name_rule_dict,"Credit_BusinessHours":Credit_BusinessHours_name_rule_dict,"Credit_OffHook":Credit_OffHook_name_rule_dict}
# nodeweight_dict = {"Credit_BWork":Credit_BWork_name_weight_dict,"Credit_OpeningRemarks2":Credit_OpeningRemarks2_name_weight_dict,"Credit_Surname":Credit_Surname_name_weight_dict,"Credit_Education":Credit_Education_name_weight_dict,"Credit_Invite":Credit_Invite_name_weight_dict,"Credit_BusinessHours":Credit_BusinessHours_name_weight_dict,"Credit_OffHook":Credit_OffHook_name_weight_dict}
# nodemap_dict = {"Credit_BWork":Credit_BWork_ename_cname_dict,"Credit_OpeningRemarks2":Credit_OpeningRemarks2_ename_cname_dict,"Credit_Surname":Credit_Surname_ename_cname_dict,"Credit_Education":Credit_Education_ename_cname_dict,"Credit_Invite":Credit_Invite_ename_cname_dict,"Credit_BusinessHours":Credit_BusinessHours_ename_cname_dict,"Credit_OffHook":Credit_OffHook_ename_cname_dict}
from Robot_Keyword_Credit import nodename_dict,nodeweight_dict,nodemap_dict,KeyWordMatcher
def Auto_Sort_Keyword_Semantic_V21(c_node_name,query):
    # 将中文节点名 ==>> 对应的英文节点名
    node_name = CNodeName_ENodeName_Dict.get(c_node_name)
    logging.info("ZH_Node_Name:{},EN_Node_Name:{},Query:{}".format(c_node_name,node_name,query))
    query = clean(query)
    prefix_matcher_list_max = 0
    matcher_bool_hitnode = None
    matcher_bool_hitnodename = None
    score_max = 0
    for key,value in nodename_dict[node_name].items():
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list,prefix_matcher_str_len = KeyWordMatcher(human_str = query,regex_str = value)
        tongpe_value = nodeweight_dict[node_name].get(key)
        if matcher_bool:
            score_current = (prefix_matcher_str_len + int(tongpe_value)) / len(query)
        else:
            score_current = (prefix_matcher_str_len) / len(query)
        if prefix_matcher_str_len > prefix_matcher_list_max:
            if score_current > score_max:
                score_max = score_current
                matcher_bool_hitnode = nodemap_dict[node_name].get(key)
                matcher_bool_hitnodename = value
    if score_max >= 1.0:
        score_max = 1.0
    
    # 规则类的匹配DataFrame:[score,hit_node,match_query]
    df_list = []
    df_list.append(pd.DataFrame({"score":score_max,"hit_node":matcher_bool_hitnode,"match_query":matcher_bool_hitnodename},index = [0]))
    # 粗召回+精排序DataFrame:[score,hit_node,match_query]
    FineSort = Fine_Sort_V2(node_name=node_name,query=query)
    # SBERT
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortSbert.iloc[0,-1],"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},index = [1]))
    # WORD2VEC
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortW2V.iloc[0,-1],"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]},index = [2]))
    res_df = pd.concat(df_list, ignore_index=True)
    result = res_df.sort_values(by=['score'],ascending=False).head(1)
    finesort_intent_score = result.iloc[0,-3] # 精排最终得分
    finesort_intent_node = result.iloc[0,-2] # 精排命中节点
    finesort_intent_match_query = result.iloc[0,-1] # 精排命中关键词集

    res_list = [
        {"rule_score":str(score_max),"hit_node":matcher_bool_hitnode,"match_query":matcher_bool_hitnodename},
        {"sbert_score":str(FineSortSbert.iloc[0,-1]),"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},
        {"w2v_score":str(FineSortW2V.iloc[0,-1]),"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]}
    ]
    return {"match_node":finesort_intent_node,"match_query":finesort_intent_match_query,"match_score":str(finesort_intent_score),"detail":res_list}

# 自研规则算法
def Auto_Sort_Rule_Semantic_V1(query):
    query = clean(query)
    from Robot_Regex import ip
    df_list = []
    # 自研规则算法解析query得分
    (intent_name,intent_score,intent_positive_rule,intent_negative_rule,entity) = ip.calculate_intent(query=query)
    df_list.append(pd.DataFrame({"score":intent_score,"hit_node":intent_name,"match_query":intent_positive_rule},index = [0]))
    # sbert算法解析query得分
    FineSort = Fine_Sort_V1(query=query)
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortSbert.iloc[0,-1],"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},index = [1]))
    # word2vec算法解析query得分
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortW2V.iloc[0,-1],"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]},index = [2]))
    res_df = pd.concat(df_list, ignore_index=True)
    result = res_df.sort_values(by=['score'],ascending=False).head(1)
    finesort_intent_score = result.iloc[0,-3]
    finesort_intent_node = result.iloc[0,-2]
    finesort_intent_match_query = result.iloc[0,-1]

    rule_score = intent_score
    sbert_score = FineSortSbert.iloc[0,-1]
    w2v_score = FineSortW2V.iloc[0,-1]

    return [finesort_intent_node, finesort_intent_match_query, rule_score, sbert_score, w2v_score]

# 自研规则算法
def Auto_Sort_Rule_Semantic_V2(query):
    query = clean(query)
    from Robot_Regex import ip
    df_list = []
    # 自研规则算法解析query得分
    (intent_name,intent_score,intent_positive_rule,intent_negative_rule,entity) = ip.calculate_intent(query=query)
    df_list.append(pd.DataFrame({"score":intent_score,"hit_node":intent_name,"match_query":intent_positive_rule},index = [0]))
    # sbert算法解析query得分
    FineSort = Fine_Sort_V1(query=query)
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortSbert.iloc[0,-1],"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},index = [1]))
    # word2vec算法解析query得分
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortW2V.iloc[0,-1],"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]},index = [2]))
    res_df = pd.concat(df_list, ignore_index=True)
    # dataframe:[score -- hit_node -- match_query]
    result = res_df.sort_values(by=['score'],ascending=False).head(1)
    finesort_intent_score = result.iloc[0,-3]
    finesort_intent_node = result.iloc[0,-2]
    finesort_intent_match_query = result.iloc[0,-1]

    res_list = [
        {"rule_score":str(intent_score),"hit_node":intent_name,"match_query":intent_positive_rule},
        {"sbert_score":str(FineSortSbert.iloc[0,-1]),"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},
        {"w2v_score":str(FineSortW2V.iloc[0,-1]),"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]}
    ]

    return {"match_node":finesort_intent_node,"match_query":finesort_intent_match_query,"match_score":str(finesort_intent_score),"detail":res_list}
            
if __name__ == '__main__':

    # df_test = pd.read_csv(args.sort_test_data_zhongan,sep="\t",header=None,names=["custom","intent_true","none0","none1","none2"])
    # # gbm_finesort_model = joblib.load(args.sort_lightgbm_model) 
    # # df_test['intent_predict'] = df_test.apply(lambda row:GBM_Sort_Test(row['custom'],gbm_finesort_model),axis=1)
    # # df_test[['intent_predict','hit_type']] = df_test.apply(lambda row:Sort_Semantic_Keyword_V1(row['custom']),axis=1,result_type='broadcast')
    # df_test[['intent_predict','match_query','keyword_score','sbert_score','w2v_score']] = df_test.apply(lambda row:Auto_Sort_Rule_Semantic_V1(row['custom']),axis=1,result_type='broadcast')
    # df_test.drop(columns=['none0','none1','none2']).to_csv('result/df_test.csv', mode='w', index=False)

    # from sklearn.metrics import classification_report
    # y_true = df_test['intent_true'].values.tolist()
    # y_pred = df_test['intent_predict'].values.tolist()

    # report = classifaction_report_to_csv(y_true,y_pred)
    # report.to_csv("result/df_report.csv", index= True)
    # print(report)
    Cur_CNodeName = "B工作"
    res = Auto_Sort_Keyword_Semantic_V21(Cur_CNodeName,"别一天到晚的给我打电话了！")
    print(res)
