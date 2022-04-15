# -*- coding: utf-8 -*-
import os
import args
import joblib
import pandas as pd
import args

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer(args.retrival_sbert)

# 基于hnsw的query问题的粗召回
# from Step2_Retrival_HNSW_FAISS import FAISS_WORD2VEC
# faiss = FAISS_WORD2VEC(w2v_path=args.retrival_word2vec_zhongan, data_path=args.retrival_hnsw_train_zhongan,faiss_model_path=args.retrival_faiss_word2vec_zhongan)
from Step2_Retrival_HNSW_FAISS import HNSW_SBERT
hnsw = HNSW_SBERT(sbert_model,args.retrival_hnsw_train_zhongan,args.retrival_hnsw_ef,args.retrival_hnsw_max,args.retrival_sbert_hnsw_model)
# 基于多种度量方式的计算两句子的相似度
from Sort_Similarity_Calculater import Similarity_Measure
sim_calc = Similarity_Measure()
# from Sentence_Bert_Calculater import SBertSimilarCalculate
# sbert_sim = SBertSimilarCalculate()
from Sentence_Bert_Calculater import SBertSimCalc
sbert_sim = SBertSimCalc(sbert_model=sbert_model)

# 召回topN问题及精排序
def Fine_Sort_V0(query,topN = 10):
    RoughRecall = pd.DataFrame()
    RoughRecall = RoughRecall.append(pd.DataFrame({'query': [query]*topN ,'retrieved': hnsw.search(query, topN)['custom'] , 'intent': hnsw.search(query, topN)['intent']})) #query问题粗召回
    # RoughRecall.to_csv('result/rough_recall.csv', mode='w', index=False)

    FineSort = pd.DataFrame()
    FineSort['query'] = RoughRecall['query'] #query问题
    FineSort['retrieved'] = RoughRecall['retrieved'] #query召回问题
    FineSort['intent'] = RoughRecall['intent'] #召回问题对应的回答

    # 多种度量方式计算query问题以及query召回问题的相似度
    RetrievedValue = pd.DataFrame.from_records(FineSort.apply(lambda row: sim_calc.Similarity_Calculate_ZhongAn(row['query'] , row['retrieved']), axis=1))
    FineSort = pd.concat([FineSort, RetrievedValue], axis=1)
    FineSort['finesort_row_max']= FineSort.max(axis=1)
    # 'query','retrieved','intent','lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','finesort_row_max'
    result = FineSort.sort_values(by=['finesort_row_max'],ascending=False).head(1)
    # result = FineSort.sort_values(by=['lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson'],ascending=[False,False,False,False,False,False,False,False,False,False]).head(1) #,'tfidf_cos','tfidf_eucl','tfidf_pearson'
    # print("finesort_intent_name:{0},finesort_intent_score:{1}".format(result.iloc[0,2],result.iloc[0,-1]))
    return result.iloc[0,2],result.iloc[0,-1]

# 召回topN问题及精排序
def Fine_Sort_V1(query,topN = 10):
    #query问题粗召回
    RoughRecall = pd.DataFrame.from_records({'query': [query]*topN ,'retrieved': hnsw.search(query, topN)['custom'] , 'intent': hnsw.search(query, topN)['intent']}) 
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

# 通过lightGBM有监督训练多特征排序模型
def GBM_Sort_Test(query,gbm_model):
    # 'query','retrieved','intent','lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson'
    FineSort = Fine_Sort_V1(query=query)
    columns=['lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson']
    FineSort['gbm_score'] = gbm_model.predict(FineSort[columns])
    result = FineSort.sort_values(by=['gbm_score'],ascending=False).head(1)
    finesort_intent_name = result.iloc[0,2]
    finesort_intent_score = result.iloc[0,-1]

    # 关键词命中节点字典
    keyword_hitnode_intent_dict = {} 
    # 关键词命中节点得分
    keyword_hitnode_score_dict = {}
    # 关键词命中意图及得分
    keyword_intent_score_dict = {}
    # 关键词query对应每个关键词节点命中的情况
    import jieba
    for key,value in keyword_dict_rdg.items():
        #res_matcher_bool,keyword_hit_cnt_list,keyword_hit_cnt_not_list,human_str_list = KeyWordMatcher(human_str=jieba.lcut(query),regex_str=value)
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list = KeyWordMatcher(human_str=jieba.lcut(query),regex_str=value)
        if matcher_bool:
            keyword_hitnode_intent_dict[key] = keyword_intent_dict_rdg.get(key) # {"keyword_zhongan_question":"质疑号码信息泄露"}
            keyword_hitnode_score_dict[key] = (len(prefix_matcher_str_list) - len(suffix_matcher_str_list)) / len(human_str_list)
            keyword_intent_score_dict[keyword_intent_dict_rdg.get(key)] = (len(prefix_matcher_str_list) - len(suffix_matcher_str_list)) / len(human_str_list)

    if finesort_intent_score >= 0.0:
        return finesort_intent_name
    else:
        if len(keyword_intent_score_dict) == 1:# 如果命中一个关键词节点
            return max(keyword_intent_score_dict,key = keyword_intent_score_dict.get)
        elif len(keyword_intent_score_dict) > 1 and finesort_intent_name in keyword_hitnode_score_dict.keys(): # 如果没有命中关键词节点则选取精排得分最大值结果
            return finesort_intent_name
        else: 
            return finesort_intent_name

# 先关键词匹配 若只命中一个节点则用关键词 若命中多个节点再启用语义排序
from Robot_Keyword import KeyWordMatcher,keyword_dict_rdg, keyword_intent_dict_rdg
def Sort_Semantic_Keyword_V1(query):
    # 精排query对应的名称及分值
    finesort_intent_name,finesort_intent_score = Fine_Sort_V0(query)
    # 关键词命中节点字典
    keyword_hitnode_intent_dict = {} 
    # 关键词命中节点得分
    keyword_hitnode_score_dict = {}
    # 关键词命中意图及得分
    keyword_intent_score_dict = {}
    # 关键词query对应每个关键词节点命中的情况
    import jieba
    for key,value in keyword_dict_rdg.items():
        res_matcher_bool,keyword_hit_cnt_list,keyword_hit_cnt_not_list,human_str_list = KeyWordMatcher(human_str=query,regex_str=value) #jieba.lcut(query)
        if res_matcher_bool:
            keyword_hitnode_intent_dict[key] = keyword_intent_dict_rdg.get(key) # {"keyword_zhongan_question":"质疑号码、信息泄露"}
            keyword_hitnode_score_dict[key] = (len(keyword_hit_cnt_list) - len(keyword_hit_cnt_not_list)) / len(human_str_list)
            keyword_intent_score_dict[keyword_intent_dict_rdg.get(key)] = (len(keyword_hit_cnt_list) - len(keyword_hit_cnt_not_list)) / len(human_str_list)

    if finesort_intent_score >= 0.9:
        return [finesort_intent_name,'semantic_09_higher']
    else:
        if len(keyword_intent_score_dict) == 1:# 如果命中一个关键词节点
            return [max(keyword_intent_score_dict,key = keyword_intent_score_dict.get),'keyword']
        elif len(keyword_intent_score_dict) > 1 and finesort_intent_name in keyword_hitnode_score_dict.keys(): # 如果没有命中关键词节点则选取精排得分最大值结果
            return [finesort_intent_name,'keyword_semantic']
        else: 
            return [finesort_intent_name,'semantic_09_lower']

# 优先关键词-再语义
def Sort_Keyword_Semantic_V1(query):
    from Robot_Keyword import keyword_dict_rdg, keyword_intent_dict_rdg
    import jieba
    matcher_bool_selector = False
    prefix_matcher_list_max = 0
    matcher_bool_hitnode = None
    matcher_bool_hittype = None
    for key,value in keyword_dict_rdg.items():
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list = KeyWordMatcher(human_str = jieba.lcut(query),regex_str = value)
        if matcher_bool and len(prefix_matcher_str_list) > prefix_matcher_list_max:
            matcher_bool_selector = True
            prefix_matcher_list_max = len(prefix_matcher_str_list)
            matcher_bool_hitnode = keyword_intent_dict_rdg.get(key)
            matcher_bool_hittype = "keyword"

    if not matcher_bool_selector:
        # 'query','retrieved','intent','lcs_score','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','sbert_score'
        FineSort = Fine_Sort_V1(query=query)
        columns=['lcs_score','edit_dist','diff_score','jaccard','w2v_cos','sbert_score']
        FineSort['finesort_row_max']= FineSort[columns].max(axis=1)
        # 'query','retrieved','intent','lcs_score','edit_dist','diff_score','jaccard','w2v_cos','sbert_score','finesort_row_max'
        result = FineSort.sort_values(by=['finesort_row_max'],ascending=False).head(1)
        matcher_bool_hitnode = result.iloc[0,2]
        matcher_bool_hittype = "semantic"
        finesort_intent_score = result.iloc[0,-1]

    return [matcher_bool_hitnode,matcher_bool_hittype]

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

# 优先关键词-再语义
def Auto_Sort_Keyword_Semantic_V1(query):
    query = clean(query)
    from Robot_Keyword import keyword_dict_rdg, keyword_intent_dict_rdg
    matcher_bool_selector = False
    prefix_matcher_list_max = 0
    matcher_bool_hitnode = None
    matcher_bool_hitnodename = None
    matcher_bool_hittype = None
    score_max = 0
    for key,value in keyword_dict_rdg.items():
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list,prefix_matcher_str_len = KeyWordMatcher(human_str = query,regex_str = value)
        score_current = (prefix_matcher_str_len + int(len(query)/2)) / len(query)
        if score_current > score_max:
            score_max = score_current
            matcher_bool_hitnode = keyword_intent_dict_rdg.get(key)
            matcher_bool_hitnodename = key
    
    df_list = []
    df_list.append(pd.DataFrame({"score":score_max,"hit_node":matcher_bool_hitnode,"match_query":matcher_bool_hitnodename},index = [0]))
    FineSort = Fine_Sort_V1(query=query)
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortSbert.iloc[0,-1],"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},index = [1]))
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortW2V.iloc[0,-1],"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]},index = [2]))
    res_df = pd.concat(df_list, ignore_index=True)
    print(res_df)
    result = res_df.sort_values(by=['score'],ascending=False).head(1)
    finesort_intent_score = result.iloc[0,-3]
    finesort_intent_node = result.iloc[0,-2]
    finesort_intent_match_query = result.iloc[0,-1]

    return [finesort_intent_node,finesort_intent_match_query,finesort_intent_score]

# 关键词奖惩(通配权值) + 语义评估(sbert + word2vec)
def Auto_Sort_Keyword_Semantic_V2(query):
    query = clean(query)
    from Robot_Keyword import keyword_dict_rdg, keyword_dict_tongpei_rdg,keyword_intent_dict_rdg
    prefix_matcher_list_max = 0
    matcher_bool_hitnode = None
    matcher_bool_hitnodename = None
    score_max = 0
    for key,value in keyword_dict_rdg.items():
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list,prefix_matcher_str_len = KeyWordMatcher(human_str = query,regex_str = value)
        tongpe_value = keyword_dict_tongpei_rdg.get(key)
        if matcher_bool:
            score_current = (prefix_matcher_str_len + int(tongpe_value)) / len(query)
        else:
            score_current = (prefix_matcher_str_len) / len(query)
        if prefix_matcher_str_len > prefix_matcher_list_max:
            if score_current > score_max:
                score_max = score_current
                matcher_bool_hitnode = keyword_intent_dict_rdg.get(key)
                matcher_bool_hitnodename = value
    if score_max >= 1.0:
        score_max = 1.0
    
    df_list = []
    df_list.append(pd.DataFrame({"score":score_max,"hit_node":matcher_bool_hitnode,"match_query":matcher_bool_hitnodename},index = [0]))
    FineSort = Fine_Sort_V1(query=query)
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortSbert.iloc[0,-1],"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},index = [1]))
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortW2V.iloc[0,-1],"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]},index = [2]))
    res_df = pd.concat(df_list, ignore_index=True)
    result = res_df.sort_values(by=['score'],ascending=False).head(1)
    finesort_intent_score = result.iloc[0,-3]
    finesort_intent_node = result.iloc[0,-2]
    finesort_intent_match_query = result.iloc[0,-1]

    keyword_score = score_max
    sbert_score = FineSortSbert.iloc[0,-1]
    w2v_score = FineSortW2V.iloc[0,-1]

    return [finesort_intent_node, finesort_intent_match_query, keyword_score, sbert_score, w2v_score]

# 关键词增加奖励项：len(query)/2
def Auto_Sort_Keyword_Semantic_V3(query):
    query = clean(query)
    from Robot_Keyword import keyword_dict_rdg, keyword_dict_tongpei_rdg,keyword_intent_dict_rdg
    prefix_matcher_list_max = 0
    matcher_bool_hitnode = None
    matcher_bool_hitnodename = None
    score_max = 0
    for key,value in keyword_dict_rdg.items():
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list,prefix_matcher_str_len = KeyWordMatcher(human_str = query,regex_str = value)
        tongpe_value = keyword_dict_tongpei_rdg.get(key)
        if matcher_bool:
            score_current = (prefix_matcher_str_len + int(len(query)/2)) / len(query)
        else:
            score_current = (prefix_matcher_str_len) / len(query)
        if prefix_matcher_str_len > prefix_matcher_list_max:
            if score_current > score_max:
                score_max = score_current
                matcher_bool_hitnode = keyword_intent_dict_rdg.get(key)
                matcher_bool_hitnodename = value
    if score_max >= 1.0:
        score_max = 1.0
    
    df_list = []
    df_list.append(pd.DataFrame({"score":score_max,"hit_node":matcher_bool_hitnode,"match_query":matcher_bool_hitnodename},index = [0]))
    FineSort = Fine_Sort_V1(query=query)
    FineSortSbert = FineSort[['query','retrieved', 'intent','sbert_score']].sort_values(by=['sbert_score'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortSbert.iloc[0,-1],"hit_node":FineSortSbert.iloc[0,-2],"match_query":FineSortSbert.iloc[0,-3]},index = [1]))
    FineSortW2V = FineSort[['query','retrieved', 'intent','w2v_cos']].sort_values(by=['w2v_cos'],ascending=False).head(1)
    df_list.append(pd.DataFrame({"score":FineSortW2V.iloc[0,-1],"hit_node":FineSortW2V.iloc[0,-2],"match_query":FineSortW2V.iloc[0,-3]},index = [2]))
    res_df = pd.concat(df_list, ignore_index=True)
    result = res_df.sort_values(by=['score'],ascending=False).head(1)
    finesort_intent_score = result.iloc[0,-3]
    finesort_intent_node = result.iloc[0,-2]
    finesort_intent_match_query = result.iloc[0,-1]

    keyword_score = score_max
    sbert_score = FineSortSbert.iloc[0,-1]
    w2v_score = FineSortW2V.iloc[0,-1]

    return [finesort_intent_node, finesort_intent_match_query, keyword_score, sbert_score, w2v_score]

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

    df_test = pd.read_csv(args.sort_test_data_zhongan,sep="\t",header=None,names=["custom","intent_true","none0","none1","none2"])
    # gbm_finesort_model = joblib.load(args.sort_lightgbm_model) 
    # df_test['intent_predict'] = df_test.apply(lambda row:GBM_Sort_Test(row['custom'],gbm_finesort_model),axis=1)
    # df_test[['intent_predict','hit_type']] = df_test.apply(lambda row:Sort_Semantic_Keyword_V1(row['custom']),axis=1,result_type='broadcast')
    df_test[['intent_predict','match_query','keyword_score','sbert_score','w2v_score']] = df_test.apply(lambda row:Auto_Sort_Rule_Semantic_V1(row['custom']),axis=1,result_type='broadcast')
    df_test.drop(columns=['none0','none1','none2']).to_csv('result/df_test.csv', mode='w', index=False)

    from sklearn.metrics import classification_report
    y_true = df_test['intent_true'].values.tolist()
    y_pred = df_test['intent_predict'].values.tolist()

    report = classifaction_report_to_csv(y_true,y_pred)
    report.to_csv("result/df_report.csv", index= True)
    print(report)