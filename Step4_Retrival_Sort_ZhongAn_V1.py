# -*- coding: utf-8 -*-
import os
import args
import joblib
import pandas as pd
import args

# 基于hnsw的query问题的粗召回
from Step2_Retrival_HNSW_FAISS import FAISS
faiss = FAISS(w2v_path=args.retrival_word2vec_zhongan, data_path=args.retrival_hnsw_train_zhongan,faiss_model_path=args.retrival_faiss_model_zhongan)
# 基于多种度量方式的计算两句子的相似度
from Sort_Similarity_Calculater import Manual_Similarity
manual_Sim = Manual_Similarity()
# 基于pytorch训练的Bert微调模型预测两句子的相似度
# from Bert_Similarity_Calculater import BertSimilarCalculate
# bert_Sim = BertSimilarCalculate()

# 召回topN问题及精排序
def Fine_Sort_V0(query,topN = 10):
    RoughRecall = pd.DataFrame()
    RoughRecall = RoughRecall.append(pd.DataFrame({'query': [query]*topN ,'retrieved': faiss.search(query, topN)['custom'] , 'intent': faiss.search(query, topN)['intent']})) #query问题粗召回
    # RoughRecall.to_csv('result/rough_recall.csv', mode='w', index=False)

    FineSort = pd.DataFrame()
    FineSort['query'] = RoughRecall['query'] #query问题
    FineSort['retrieved'] = RoughRecall['retrieved'] #query召回问题
    FineSort['intent'] = RoughRecall['intent'] #召回问题对应的回答

    # print(FineSort)
    # 多种度量方式计算query问题以及query召回问题的相似度
    RetrievedValue = pd.DataFrame.from_records(FineSort.apply(lambda row: manual_Sim.Similarity_Calculate_ZhongAn(row['query'] , row['retrieved']), axis=1))
    # print(RetrievedValue)
    FineSort = pd.concat([FineSort, RetrievedValue], axis=1)
    # FineSort['bert_score'] = FineSort.apply(lambda row: bert_Sim.predict(row['question1'] , row['question2'])[1], axis=1)
    FineSort['finesort_row_max']= FineSort.max(axis=1)
    # 'query','retrieved','intent','lcs','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','finesort_row_max'
    result = FineSort.sort_values(by=['finesort_row_max'],ascending=False).head(1)
    # print(result)
    # result = FineSort.sort_values(by=['lcs','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson'],ascending=[False,False,False,False,False,False,False,False,False,False]).head(1) #,'tfidf_cos','tfidf_eucl','tfidf_pearson'
    # print("finesort_intent_name:{0},finesort_intent_score:{1}".format(result.iloc[0,2],result.iloc[0,-1]))
    return result.iloc[0,2],result.iloc[0,-1]

# 召回topN问题及精排序
def Fine_Sort_V1(query,topN = 10):
    RoughRecall = pd.DataFrame()
    RoughRecall = RoughRecall.append(pd.DataFrame({'query': [query]*topN ,'retrieved': faiss.search(query, topN)['custom'] , 'intent': faiss.search(query, topN)['intent']})) #query问题粗召回
    # RoughRecall.to_csv('result/rough_recall.csv', mode='w', index=False)

    FineSort = pd.DataFrame()
    FineSort['query'] = RoughRecall['query'] #query问题
    FineSort['retrieved'] = RoughRecall['retrieved'] #query召回问题
    FineSort['intent'] = RoughRecall['intent'] #召回问题对应的回答

    # print(FineSort)
    # 多种度量方式计算query问题以及query召回问题的相似度
    RetrievedValue = pd.DataFrame.from_records(FineSort.apply(lambda row: manual_Sim.Similarity_Calculate_ZhongAn(row['query'] , row['retrieved']), axis=1))
    # print(RetrievedValue)
    FineSort = pd.concat([FineSort, RetrievedValue], axis=1)
    # FineSort['bert_score'] = FineSort.apply(lambda row: bert_Sim.predict(row['question1'] , row['question2'])[1], axis=1)
    FineSort['finesort_row_max']= FineSort.max(axis=1)
    # 'query','retrieved','intent','lcs','edit_dist','diff_score','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','finesort_row_max'
    res = FineSort[FineSort['query','retrieved','intent','finesort_row_max']].sort_values(by=['finesort_row_max'],ascending=False).head(3) # 返回Top3的DataFrame
    return res

# 通过classifaction_report生成相应结果报告
def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        data = line.split()
        if len(data) == 5:
            row['class'] = data[0]
            row['precision'] = data[1]
            row['recall'] = data[2]
            row['f1_score'] = data[3]
            row['support'] = data[4]
            report_data.append(row)
    return pd.DataFrame.from_dict(report_data)

# 节点1-质疑
keyword_zhongan_question = "((怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话)|((谁给|谁告诉|哪里|得到|拿到|搞到|哪来|哪搞的|哪里来|哪里拿|为什么你们知道|为什么你知道|为什么你有|为什么你们有|为什么知道|怎么知道|怎么会有|怎么有|怎么得|怎么来|怎么拿到)&(电话|手机|号码|信息|联系方式)))&~(手机号有|手机里有|手机有)"
# 节点2-肯定
keyword_zhongan_yes = "((嗯|恩|现在就去|噢)|(发|对|好|红|宏|好的|好得|好啊|好呀|行|兴|星|型|性|形|姓|邢|鑫|行呀|行啊|行吧|号|浩|豪|昊|耗|毫|郝的|照|兆|可以|要得|想搞|好好|好吧|那行|那好|行行|有有|OK|YES|ok|yes|Yes|对的|对啊|中意的|中啊|弄个|整个|弄一个|试一下|办一个|整一个|不错|挺好|对呀|对哦|对对|对啊|噩耗|哦好)|(可以|开呀|春雨|和一|科技|和e|e一|尅一|何一|可怜|康定|康莉|开业|可意|我k|副业|枯叶|开呀|海底|可惜|何以|可疑|刻意|科一|可疑|开机|可谓|荷叶|开一|和谐|可喜|和颐|荷叶|可能也|和田|合理|可以考虑|完全可以|你说|看一下|介绍|你讲|看看)|(想|想啊|想呀|想哦|箱包|想听|想要|准备)|(要|要的|要啊|要呀|我要|须要|需要|徐尧|需求|正好需要|方便|可能会)|(愿意|没问题|当然|确定|毫无疑问|淳意|同意|统一|同一|同义|同业|藤椅)|(么问题|没问题|没问提|每一位提|没有问题)|(有兴趣|感兴趣)|(有的|有需要|有想法|正有想法|正好有想法|正好|来的好巧)|((邮|有|又|油|由|友|右)&(信使|兴趣|信息|兴趣|献曲|性趣|行去|需要))|(是得|是呀|当然有|是的|是哎|是哦|似的|是啊|试了|施了|世道|士的|试的|式的|失当|使得|石槽|事得|叙了|胜过|生活|圣德|小岛|使得|吕的|世界|适当|十等|是掉)|(妨便|放鞭|翻遍|方遍|仿编|又时间|方便|噢遍|放便|有时间|有空)|(明白|知道|听明白了|了解|了姐|乐姐|勒姐|晓得)|(发过来|给过来|弄过来|整过来|给我发短信|加微信|加联系方式)|(清楚|收到)|(手头紧|紧张)|(微信|微心|威信|维新|违心|短信)|((发)&(信息|资料))|(发个信息|发条信息|发我信息|发条信息给我|发资料|发资料吧|我只看看资料|发资料我看下|你们有什么资料吗|发地址|发个地址|发定位|发给我个网址|发个联系方式给我|发我下|发给我|直接发吧|发我看一下|先发给我看看|发到手机|发到我手机上|发短信|发个短信|发我短信|给我)|(给我打过来|发过去|链接|邮件|邮箱|发过来|发给我|发资料|发一个|发一条|发过来|发给我|发条|发吧|发啊|发呀|发一份|发个|发份|让他联系我|帮我约|怎么联系|要过两天|怎么找你|给个联系|怎么跟你联系|怎么联系|留个联系|留个电话|打你电话|看看|看一下|微信|维修|威信|卫星|微型|微星|短信|段鑫|段欣|信息|欣喜|违心|维新|唯心|心细|心系|新戏|心喜|断心|断新|端信|端心|短消息|短效西|短小溪|短小戏|段小溪|段小戏|断消息|断小西|断小戏)|((加|发|罚|伐|法)&(链接|连接|廉洁|联结|莲姐|邮箱|邮件|手机|微信|维修|威信|威信|卫星|微型|微星|资料|信息|短信)))&~(不要|不知道|不行|不明白|可以吗|知道吗|明白吗|好吗|号码|你好|您好|号线|警号|新号|信号|有了|你说什么|不要了|你好了吧|我为什么要借|可以你)"
# 节点3-考虑
keyword_zhongan_consider = "((下个月|还有几个月|过段时间|考虑|考虑一下|考虑考虑|思考|考一下|考量|寻思|巡视|想想|商量|再讲|再谈|再聊|想一下|想一想|向一下|再说|在说|有时间看看|回头再说)|((等|到|要|看)&(晚上|下午|明天|下班)))&~(嗯|行|好|嘞)"
# 节点4-不方便
keyword_zhongan_inconvenient = "((在忙|好忙|现在忙|三毛|三忙|都很忙|生意忙|有点忙|我很忙|湛蓝|戴吗|嗯忙|在囊|下马|在骂|干活|急事|做事|有事|幼师|我有事|有事忙|有点事|有电视|优点是|事情特别多|事情很多|实情特别夺|实情很夺|事很忙|事很多|有点忙|等一下|忙的很|忙得很|再说吧|不得空|没有功夫|忙得很)|(开车|开扯|凯车|开查|开插|开着车|高速|在车上|倒车|骑车|弃车|上班|在工作|开会|开回|辉凯|开慧|凯辉|火车上|在路上)|(睡觉|这叫|短信叫|但是较|再生胶|这学校|谢谢你叫|在学校|在这叫|暂时将|暂时讲|午休|无袖|无休|休息|修习|修细|修鞋|的时候|的寿喜|在响水|在酒席|大兴西|戴小姐|大小姐|包修吧|在游戏|再说谢|在纠结|在就讲|在消息|再就业|在纠结|在相信|短消息|封丘县|段小姐)|(不方便|五方便|富邦店|不尝遍|不帮明|不方遍|不放便|不通电|不妨便|不商店|不广电|不放鞭|不翻遍|不仿编|布方便|五方便|不帮明|不方遍|不放便|不通电|补方便|多变)|((不|部|步|补|布|埠)&(通电|房里|妨便|商店|广电|观点|回页|关键|尝遍|上电|当面|双倍|放鞭|翻遍|方遍|仿编|方便|噢遍|上边|常见|长店|反比|帮变|黄片|分辨|燕|帮边|帮别|帮明|传店|返航店|芳店|画面|黄店|宽甸|状元|返返|放便|上面|三点的|芳甸|看扁|后面|有扣|交扣|上吧|双倍|常见|长店|反比|帮变|黄片|分辨|话费|帮听|夜空|铀矿|疏通|恩慧))|(没空|恶魔时间|美食|慢食街|没实践|没闲|没时间|没事掉|卫生间)|((现在|这会|这伙)&(没空|么空|木时间|莫时间|没时间|没有时间|木有时间|抹油时间|莫有时间|忙))|(哪有时间|磨时间|魔时间|没空|慕课|没四溅|没死间|么得时间|没得时间|莫时间|莫空|莫得空|么得空|买的空|没得空|蓦地空|没得闲|美德鲜|没得先|妹的鲜|美的线|确实没时间|缺什么时间|没呦时间|不确定时间)|((没|美|煤|每|眉|妹|莫|没有|没哟|煤油|没用|没油|木有|梅|木|霉|味)&(控|恐|空|时间|实践|事件|世间))|(电梯里|点题|电提|电机里|见街李|见经理|键精灵|二电器|剑经理|店七里|电机了|电街李|电天理|见挺立|见清理|天皮影|见铁岭|电蹄岭|殿西里|在舰艇|再见厅里|在建平里|建厅里|半天机里|我在电信|再见京山县|豆浆机里|在边境|见基地|稍等)|(明年再说|再见|拜拜|算了|酸了|算嘞|算啦|酸辣|再联系|在联系|在聊|再聊|在说|再说|先不弄|晚点联系|网一点|王店|晚点|晚一点|玩点|万店|网点|网店|王典|以后再说|一吼|等下|登下|再说|灯下|在说吧|以后再说|蚁后|有需要联系|有需要再联系|有需要在联系|等下联系|晚上联系|好像联系|等一下联系|回家联系|不要联系|本上联系|人家联系|帮下联系|上家园店|本家勉县|本上林县|盛源西|请假联系|本膳联系|网站联系|本下月去|等下也去|你下载联系|沈家诚信|上下联信|史家零是|本校的是|北票联系|本号联系|手下联系|到下联系|一下联系|刚下联系|对外联系|的话联系|晚上你行|等下梁溪|等下甸县)|((晚点|完电|万里|玩点|王店|玩店|完全|完年|网点|兰店|完整|王京|难点|关键|王明|门店|观点|怀念|完吊|完点|关你|我内|我能|信号不好|定好不好|不老不好|你好不好|七号不好|什邡不好|金好不好|线路不好|建好不好|幸好不好|亲浩不好|请好不好|最后不好|等下|等会|网一点|王店|晚点|晚上|等一下|好像|回家|不要联系|本上|本下|帮下|人家|上家|本家|请假|你下|沈家|上下|一下|刚下|的话|回头)&(再打|打|大|在打|在八|真的|在这|的了|的吧|代码|在大|东大|短打|在的|在党|别的|再说|办的|再到|较大|菜|得到|带吧|代办|带吧|针打|联系|游戏|梁溪))|(等一年|等两年|等一段时间|过段时间|过一段时间|不一定有时间|等一下|等会|等下|蚁后|再说|在说|再讲|以后再说|再看)|((到时候|不一定|不确定|有时间)&(在看|看一看|看一下|谈一谈|有空|就去))|((等下|等会|网一点|王店|晚点|晚上|等一下|好像|回家|不要|一下|刚下|稍后|稍候)&(联系|游戏|梁溪|练习|席))|((有空|有时间|晚点|迟点|回头|待会|过会|一会|等会|过一会|稍等|等等|晚上|下午|明天|下班|下课|放学|上完课|星期六|礼拜六|星期天|礼拜天|周六|周日|周末|下星期|下周|寒假|暑假)&(再说|确认))|((有空|有时间|晚点|迟点|回头|待会|过会|一会|等会|过一会|稍等|等等|晚上|下午|明天|下班|下课|放学|上完课|星期六|礼拜六|星期天|礼拜天|周六|周日|周末|下星期|下周|寒假|暑假)&(联系|过来|打电话|打我电话|打过来|打给我))|((再)&(打|电话))|(没有电|没电|我信心定了|我手机都不要了)|((信号|新号|心好)&(不好|差|不行))|((到时候|不一定|不确定)&(时间)))&~((有事么|有事吗|没上班|没有上班)|((再说|再讲|在说|退回去)&(一遍|考虑)))"
# 节点5-拒绝
keyword_zhongan_no = "((不要|算了|没有|么有|木有|不用|不买|不好|不可以|不行|不想要|不辅导|不服的)|((没|没有|木有|木|莫|莫有|抹油|恶魔)&(用|兴趣|需要|需求|考虑|了解|了姐|乐姐|意向))|(不感兴趣|不想了解|不用介绍|不想要|不想了解|没有钱|先这样|没有兴趣|用不到|没效果|不参加|不想听|没必要|就这样吧|不想要|要不着|没考虑|用不到|用不着|用不上|没心情|没钱|很穷|没兴趣|不同意|不接受|不研究|不好意思|不了解|不需|不想|不考虑|不要|不用|不需要|不须要|不打算|没想|没有想|五八零|不八零|不报了|够了|扣了|不忘|么信使|么兴趣|么信息|么兴趣|么兴去|么献曲|么谢谢|么性趣|么行去|么需要|么需求|么打算|么必要|么想|恶魔|你打算|用不到|用不着|用不上|用不了|跟我没关系|和我没关系|哦没关系|没说要办|没说要报|墨需要|不弄|不行|不干|不怕|不用|不好|不办|补办|不了|不对|不可能|不要|不准备|不闹|别闹|不玩|不弄|不高|不搞|不做|不成|不整|不打算|没打算|没考虑|没准备|顾不上|我不是他|没进去|每兴趣|没献曲|没谢谢|莫兴趣|莫姓钱|为新区|恶魔兴趣|魔兴趣|有事情|来兴趣|北新区|姓恶魔太兴趣|齐美|不不|那先这样)|((没有|煤油|没油|美柚|美优|美油|美游|每|梅|美|煤|霉|没|其他|莫|么有)&(信使|兴趣|信息|兴趣|兴去|献曲|谢谢|性趣|行去|需要|需求|打算|必要))|(不信你|不相信|不信|不懂)|(不差钱|我有钱|不差几号|不差钱|不缺钱|有的是钱|不需要借钱了|借到钱了|我给你钱|我不要钱|我给你要不要|埋汰|我不想听|我不办|我为什么要借|你好了吧|不要|不知道|不行|你好了吧|可以你|不还可以吗)|(同行|通城|通航|同航|同一个行业|同一行|通函)|((自己|我)&(是|做|也做)&(这个|贷款|带宽|带款|借款|结款|解款|接款))|((也是|夜市|夜视|也使|野史|我比你|宾利|吉米)&(贷款|带宽|带款|借款|结款|解款|接款|行业))|((贷款|带宽|带款|借款|结款|解款|接款)&(从业者|上班|从事))|((家人|嫁人|加人|佳人|假人|嘉人|佳仁|夹人|朋友|彭友|兄弟|熊迪|家属|家书|家数|加数|家鼠|家塾|老公|老婆|紫女|子女|父母|父亲|母亲|对象|媳妇|哥|弟|姐|妹|爸|妈|兄弟|本身|我是|我就是|我也是|我都是|自己是|自己就是|自己都是|我这是)&(这行|干这个|弄这个)))&~(怎么|可不可以|是不是|有没有|没听清|没有听清楚|没有听明白|没有时间|还没想好|好办|好不好|你再说|你给我|在忙|再说吧|信号不好|看看再说|人工打电话|发过来|要不要利息|这个也行|抵押|能再说|费用|噢行|嗯行|嗯好吗|你发吧|可以你发|可以你有)"
# 节点6-投诉
keyword_zhongan_complaint = "((又是这种电话|挂了|别打了|不要再打|不要再给我打|别给我电话|不要给我打电话|不要再给我电话|还打电话|半夜|几点|打过了|打错|找错人|打出了|打错了|搞错了|错了|又打|有打|老打|老是打|一直打|老是给我打|老给我打|老师给我打|还打|别打|别再|别在|别给我打|不要打|不要再打|不要在打|联系过|不要联系|不要再联系|别再|别人来了|你别下来了|乱打)|(听懂人话|听不懂人话|烦不烦|返不返|山不返|无聊|很烦|挺烦|停放|好烦|天天|甜甜|舔舔|太烦了|烦人|应凡|全部烦|坑死了|疼死了|痛死了)|(拨打的|订餐号码|已关机|稍后再拨|请按|请阿姨|按一|挂机|空号|扣号|请挂机|无法接通|正在通话中)|(骚扰|搔扰|搔饶|慅扰|扫绕|嫂绕|扫扰|骚饶|骚绕|扰民|饶敏|来算了)|(病|并|冰|兵|有病|有毛病|神经病|油饼|草泥马|妈的|神经|深井|沈静|肾经|油饼|有兵|有冰|有电吧|有斌吧|毛病|猫病|吃饱|发癫|脑子坏|脑子不好|傻|傻子|傻逼|煞笔|沙比|白痴|白吃|百尺|混蛋|还贷|环蛋|魂淡|浑蛋|去死|屁|劈|流氓|榴芒|刘忙|无赖|不赖|诬赖|无来)|(妈个逼|妈逼|不懂问你妈|你妈|大爷|妈卖批|卖逼|骂你|你逼|曹尼玛|操你妈|草拟|草你|操你|你妈|尼玛|他妈|妈勒个逼|你吗|接你吗|我要你大|你天天累不累)|(举报|几吧|聚宝|菊爆|聚爆|拒保|巨宝|具保|局帮|取包|几包|君帛|曲阜|取暴|警报|蕖报|请吧|去办|取帮|就吧|纸包|军办|绝望|曲吧|去抱|取败|取吧|军包|救包|季报|鸡煲|曲弯|鸡煲|几道|几万|取哇|七八|西鲍|去帮|趣鲍|当中波吧|蕖帮|迟报)|(投诉|投宿|偷诉|头诉|起诉|头苏宁|诉讼|同祝|头苏|泣诉|吸塑|起说|起脱|娶说|启动|取树|青树|体是|体收|挤出|器说|去做|期驻|曲说|期说|从你们|结束|骑术|欺负|解数|技术|给付|刑诉|漆树|调素)|(骗子|忽悠|骗人|缺德|缺的|骗鬼|诈骗|骗|片|瞎说|胡说|乱说|乱讲|瞎扯|不靠谱)|(曝光|报光|帮我广|刨工|保关|报关|报挂|报销媒体|曝光媒体|大众曝光|打中刨工|刨工|打中|打中曝光|大众报挂|到账曝光)|(110|咬咬乐|药幺零|妖妖灵|妖妖零|幺零)|(报警|报案|宝景|绊脚|包景|帮我景|包茎|饱经|报警|保靖|宝镜|宝井|包茎|报警抓我|抱起刷我|井查|姓查|经查|警告|净高|敬告|警察|检查|金莎|姓查|经常|遥岭|清查|紧查|请查|井茶|巡查)|(我想见你|我想接你|调戏|给我唱个歌)|(不要不要|不可以不可以|不行不行)|(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话)|((谁给|谁告诉|哪里|得到|拿到|搞到|哪来|哪搞的|哪里来|哪里拿|为什么你们知道|为什么你知道|为什么你有|为什么你们有|为什么知道|怎么知道|怎么会有|怎么有|怎么得|怎么来|怎么拿到)&(电话|手机|号码|信息|联系方式))|(口碑差|口碑太差|好差|口碑不好|口碑坏|坏得很|坏的很|不合规|不合法|不讲法|讲法))&~(((再说|再讲|在说)&(一遍))|(手机号有|手机里有|手机有|我的手机号就是|发到我|嗯行))"

# 机器人关键词匹配算法
def KeyWordMatcherVer2(human_str,regex_str):
    regex_res = []
    regex_tmp = []
    keyword_hit_list = []
    for regex_char in iter(regex_str):
        if regex_char == "(" or regex_char == ")" or regex_char == "|" or regex_char == "&":
            if len(regex_tmp) > 0:
                if "".join(regex_tmp) in human_str:
                    regex_res.append("True")
                    keyword_hit_list.append("".join(regex_tmp))
                else:
                    regex_res.append("False")
                regex_tmp.clear()
            regex_res.append(regex_char)
        else:
            regex_tmp.append(regex_char)

    if len(regex_tmp) > 0:
        if "".join(regex_tmp) in human_str:
            regex_res.append("True")
            keyword_hit_list.append("".join(regex_tmp))
        else:
            regex_res.append("False")
    return "".join(regex_res),set(keyword_hit_list)

def KeyWordMatcherMain(human_str,regex_str):
    split_str = regex_str.split("&~")
    if len(split_str) == 2:
        yes_matcher_res_bool,yes_matcher_res_list = KeyWordMatcherVer2(human_str=human_str,regex_str=split_str[0])
        no_matcher_res_bool,no_matcher_res_list = KeyWordMatcherVer2(human_str=human_str,regex_str=split_str[1])
    return eval(yes_matcher_res_bool+"&~"+no_matcher_res_bool),yes_matcher_res_list,no_matcher_res_list,set(human_str)


# 关键词命中节点词典
keyword_dict = {"keyword_zhongan_question":keyword_zhongan_question,"keyword_zhongan_yes":keyword_zhongan_yes,"keyword_zhongan_consider":keyword_zhongan_consider,"keyword_zhongan_no":keyword_zhongan_no,"keyword_zhongan_inconvenient":keyword_zhongan_inconvenient,"keyword_zhongan_complaint":keyword_zhongan_complaint}
# 节点与对应label的映射词典
intent_mapping_dict = {"keyword_zhongan_question":"质疑号码、信息泄露","keyword_zhongan_yes":"肯定","keyword_zhongan_consider":"同意配合","keyword_zhongan_no":"不需要","keyword_zhongan_inconvenient":"普通忙","keyword_zhongan_complaint":"厌恶"}

def KeyWord_Sort_V1(query):
    import jieba
    # 关键词query对应每个关键词节点命中的情况
    keyword_intent_list = []
    keyword_ishit_list = []
    keyword_hitwordcnt_list = []
    keyword_unhitwordcnt_list = [] 
    for key,value in keyword_dict.items():
        res_matcher_bool,keyword_hit_cnt_list,keyword_hit_cnt_not_list,human_str_list = KeyWordMatcherMain(human_str=jieba.lcut(query),regex_str=value)
        keyword_intent_list.append(intent_mapping_dict.get(key))
        keyword_ishit_list.append(res_matcher_bool)
        keyword_hitwordcnt_list.append(len(keyword_hit_cnt_list))
        keyword_unhitwordcnt_list.append(len(keyword_hit_cnt_not_list))
    return pd.DataFrame({"intent":keyword_intent_list,"ishit":keyword_ishit_list,"hitwordcnt":keyword_hitwordcnt_list,"unhitwordcnt":keyword_unhitwordcnt_list})

def KeyWord_FineSort_V1(query):
    # 'query','retrieved','intent','finesort_row_max'
    finesort_df= Fine_Sort_V1(query)
    # 'intent','ishit','hitwordcnt','unhitwordcnt'
    keyword_df = KeyWord_Sort_V1(query)
    df = pd.merge(keyword_df,finesort_df,how="left",on="intent")
    t1_df = df[['intent',"finesort_row_max","ishit"]]
    return t1_df

    
# 先关键词匹配 若只命中一个节点则用关键词 若命中多个节点再启用语义排序
def Sort_Test(query):
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
    for key,value in keyword_dict.items():
        res_matcher_bool,keyword_hit_cnt_list,keyword_hit_cnt_not_list,human_str_list = KeyWordMatcherMain(human_str=jieba.lcut(query),regex_str=value)
        if res_matcher_bool:
            keyword_hitnode_intent_dict[key] = intent_mapping_dict.get(key) # {"keyword_zhongan_question":"质疑号码、信息泄露"}
            keyword_hitnode_score_dict[key] = (len(keyword_hit_cnt_list) - len(keyword_hit_cnt_not_list)) / len(human_str_list)
            keyword_intent_score_dict[intent_mapping_dict.get(key)] = (len(keyword_hit_cnt_list) - len(keyword_hit_cnt_not_list)) / len(human_str_list)

    if finesort_intent_score >= 0.8:
        return finesort_intent_name
    else:
        if len(keyword_intent_score_dict) == 1:# 如果命中一个关键词节点
            return max(keyword_intent_score_dict,key = keyword_intent_score_dict.get)
        elif len(keyword_intent_score_dict) > 1 and finesort_intent_name in keyword_hitnode_score_dict.keys(): # 如果没有命中关键词节点则选取精排得分最大值结果
            return finesort_intent_name
        else: 
            return finesort_intent_name

if __name__ == '__main__':
    # res = Fine_Sort_Test('噢我我考虑考虑再说吧嗯')
    # print(res)
    df_test = pd.read_csv(args.sort_test_data_zhongan,sep="\t",header=None,names=["custom","intent_true"])
    df_test['intent_predict'] = df_test.apply(lambda row:Sort_Test(row['custom']),axis=1)
    df_test.to_csv('result/df_test.csv', mode='w', index=False)

    from sklearn.metrics import classification_report
    y_true = df_test['intent_true'].values.tolist()
    y_pred = df_test['intent_predict'].values.tolist()
    # report = classification_report(y_true, y_pred, output_dict=True)
    # print(report)
    # df_report = pd.DataFrame.from_dict(report)
    # df_report.to_csv('result/df_report.csv', mode='w', index=False)
    report = classification_report(y_true, y_pred)
    df_report = classifaction_report_csv(report)
    df_report.to_csv("result/df_report.csv", index= True)

    # test_sort = pd.DataFrame.from_records(df_test.apply(lambda row:Fine_Sort_Test(row['custom']),axis=1))
    # print(test_sort)
    # query_res = pd.concat([df_test,test_sort],axis=1)
    # print(query_res)

    # FineSort['max_val'] = FineSort[['lcs','edit_dist','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','tfidf_eucl','tfidf_pearson']].max(axis=1)
    # FineSort['max_idx'] = FineSort[['lcs','edit_dist','jaccard','w2v_cos','w2v_eucl','w2v_pearson','fast_cos','fast_eucl','fast_pearson','tfidf_eucl','tfidf_pearson']].idxmax(axis=1)
    # # print(FineSort[['max_val','max_idx']])
    # result = FineSort.sort_values(by=['max_val'],ascending=False).head(1)
    # print(result[['query','retrieved','intent','max_val','max_idx']])

    # 加载lightgbm模型做精排序,精排结果结合了多种相似度计算方法&ligthgbm模型问题精排：lcs、edit_dist、jaccard、bm25、w2v_cos、w2v_eucl、w2v_pearson、w2v_wmd、fast_cos、fast_eucl、fast_pearson、fast_wmd、tfidf_cos、tfidf_eucl、tfidf_pearson、bert_score
    # GBM_FineSort_Model = joblib.load(args.sort_lightgbm_model) 
    # columns = [i for i in FineSort.columns if i not in ['query' , 'retrieved' , 'target', 'answer']]
    # FineSort['sort_score'] = GBM_FineSort_Model.predict(FineSort[columns])
    # result = FineSort.sort_values(by=['sort_score'],ascending=False)
    # result.to_csv('result/fine_sort.csv', mode='w', index=False)