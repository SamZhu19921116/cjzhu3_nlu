from RuleNER import register_type
from RuleNER import SingleRuleCreator, RuleCreator
from RuleNER import SingleIntentParse, IntentParse

keyword_dict_tongpei_rdg = {"keyword_zhongan_lowcredit":30,"keyword_zhongan_lowcredit1":15,"keyword_zhongan_lowcredit2":20,"keyword_zhongan_lowcredit3":15,"keyword_zhongan_dirty":25,"keyword_zhongan_talklater":15,"keyword_zhongan_talklater1":20,"keyword_zhongan_talklater2":20,"keyword_zhongan_alreadyhave":15,"keyword_zhongan_hate":25,"keyword_zhongan_material":15,"keyword_zhongan_material1":20,"keyword_zhongan_identity":15,"keyword_zhongan_link":15,"keyword_zhongan_company":15,"keyword_zhongan_company1":15,"keyword_zhongan_company2":15,"keyword_zhongan_company3":15,"keyword_zhongan_nothear":15,"keyword_zhongan_notreceived":15,"keyword_zhongan_notreceived1":15,"keyword_zhongan_cooperate":15,"keyword_zhongan_peer":10,"keyword_zhongan_flirt":25,"keyword_zhongan_flirt1":15,"keyword_zhongan_commonbusy":25,"keyword_zhongan_age":20,"keyword_zhongan_age1":20,"keyword_zhongan_nomoney":12,"keyword_zhongan_interest":20,"keyword_zhongan_reputation":15,"keyword_zhongan_sure":10,"keyword_zhongan_sure1":12,"keyword_zhongan_sure2":1,"keyword_zhongan_sure3":1,"keyword_zhongan_drive":25,"keyword_zhongan_black":15,"keyword_zhongan_repay":20,"keyword_zhongan_quota":25,"keyword_zhongan_quata1":25,"keyword_zhongan_loanspeed":15,"keyword_zhongan_unwanted":30,"keyword_zhongan_unwanted1":10,"keyword_zhongan_unwanted2":10,"keyword_zhongan_distrust":10,"keyword_zhongan_noparty":25}
# 质疑号码信息泄露
register_type("disclosure_positive_rule1",r"(怎么|哪里|从哪|为啥|哪个|什么渠道|哪个渠道)")
register_type("disclosure_positive_rule2",r"(知道|搞到|弄到|得到|晓得|拿到|找到)")
register_type("disclosure_positive_rule3",r"(号码|电话|联系方式)")
rc_disclosure = SingleRuleCreator("keyword_zhongan_disclosure") \
        .add_positive_rule("*{disclosure_positive_rule1:disclosure_positive_rule1}*?{disclosure_positive_rule2:disclosure_positive_rule2}*?{disclosure_positive_rule3:disclosure_positive_rule3}*") \
        .add_rule_weight(15)
# 征信低
register_type("lowcredit_positive_rule11",r"(申请失败|审核失败|没通过|申请不到|通不过|申请不了|审核不过|审核不通过|信用烂了|征信烂了|征信差劲|逾期|没有资格|不通过|申请不下来|贷不下来|批不下来|没有通过|没通过|不能通过|信用不行|征信不行)")
register_type("lowcredit_negative_rule1",r"(骗)")
rc_lowcredit = SingleRuleCreator("keyword_zhongan_lowcredit") \
        .add_positive_rule("*{lowcredit_positive_rule11:lowcredit_positive_rule11}*") \
        .add_negative_rule("*{lowcredit_negative_rule1:lowcredit_negative_rule1}") \
        .add_rule_weight(30)
register_type("lowcredit_positive_rule21",r"(征信|信用度|信用卡)")
register_type("lowcredit_positive_rule22",r"(不好|不高|不行|逾期|问题)")
rc_lowcredit1 = SingleRuleCreator("keyword_zhongan_lowcredit1") \
        .add_positive_rule("*{lowcredit_positive_rule21:lowcredit_positive_rule21}*?{lowcredit_positive_rule22:lowcredit_positive_rule22}*") \
        .add_rule_weight(15)
register_type("lowcredit_positive_rule31",r"(申请|借|贷|试了|试过|征信|通过|信誉|诚信|信用)")
register_type("lowcredit_positive_rule32",r"(不下来|下不来|没额度|没过|不过|不成|不批|不出来|出不来|不成功|不良|不出钱|过不去|不了|不好|不太好|不够)")
register_type("lowcredit_negative_rule3",r"(年龄|年纪|岁数|还不了|骗)")
rc_lowcredit2 = SingleRuleCreator("keyword_zhongan_lowcredit2") \
        .add_positive_rule("*{lowcredit_positive_rule31:lowcredit_positive_rule31}*?{lowcredit_positive_rule32:lowcredit_positive_rule32}*") \
        .add_negative_rule("*{lowcredit_negative_rule3:lowcredit_negative_rule3}") \
        .add_rule_weight(20)
register_type("lowcredit_positive_rule41",r"(不贷给我|不好贷|没有额度)")
register_type("lowcredit_negative_rule4",r"(骗|已经申请)")
rc_lowcredit3 = SingleRuleCreator("keyword_zhongan_lowcredit3") \
        .add_positive_rule("*{lowcredit_positive_rule41:lowcredit_positive_rule41}*") \
        .add_negative_rule("*{lowcredit_negative_rule4:lowcredit_negative_rule4}") \
        .add_rule_weight(15)
# 脏话骂人
register_type("dirty_positive_rule11",r"(我操|妈的|死|滚蛋|你妈|你妹|有病|他妈|鸡巴|神经病|妈个逼|脑子有问题|有毒|傻逼)")
register_type("dirty_negative_rule1",r"(操作)")
rc_dirty = SingleRuleCreator("keyword_zhongan_dirty") \
        .add_positive_rule("*{dirty_positive_rule11:dirty_positive_rule11}*") \
        .add_negative_rule("*{dirty_negative_rule1:dirty_negative_rule1}") \
        .add_rule_weight(25)
# 以后再说
register_type("talklater_positive_rule11",r"(看一下|再说|看看|再看|再联系|再会|稍后再打|再试试|看情况|以后再说)")
register_type("talklater_negative_rule1",r"(开车|发|一遍|一次|额度|忙|有事情|不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|现在忙|正忙|不方便|不太方便|上班|开会|没时间|没啥时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|不需|多少钱|利息)")
rc_talklater = SingleRuleCreator("keyword_zhongan_talklater") \
        .add_positive_rule("*{talklater_positive_rule11:talklater_positive_rule11}*") \
        .add_negative_rule("*{talklater_negative_rule1:talklater_negative_rule1}") \
        .add_rule_weight(15)
register_type("talklater_positive_rule21",r"(等|到时候|到时|的时候|以后|后面|过俩天|过几天|下个月|晚上|回去|明天|晚点)")
register_type("talklater_positive_rule22",r"(需要|有空|一下|一会|看|再弄|再用|再办|去办|再试试|试|打|商量|回复|申请)")
register_type("talklater_negative_rule2",r"(不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|现在忙|正忙|不方便|不太方便|上班|开会|没时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|开车|不需|多少钱|在外面|他)")
rc_talklater1 = SingleRuleCreator("keyword_zhongan_talklater1") \
        .add_positive_rule("*{talklater_positive_rule21:talklater_positive_rule21}*?{talklater_positive_rule22:talklater_positive_rule22}*") \
        .add_negative_rule("*{talklater_negative_rule2:talklater_negative_rule2}") \
        .add_rule_weight(20)
register_type("talklater_positive_rule31",r"(需要)")
register_type("talklater_positive_rule32",r"(再用|再办|再弄|再注册|再找你|再联系|再来|再登录)")
register_type("talklater_negative_rule3",r"(不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|现在忙|正忙|不方便|不太方便|上班|开会|没时间|没啥时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|不需|多少钱)")
rc_talklater2 = SingleRuleCreator("keyword_zhongan_talklater2") \
        .add_positive_rule("*{talklater_positive_rule31:talklater_positive_rule31}*?{talklater_positive_rule32:talklater_positive_rule32}*") \
        .add_negative_rule("*{talklater_negative_rule3:talklater_negative_rule3}*") \
        .add_rule_weight(20)
# 已经有了
register_type("alreadyhave_positive_rule11",r"(申请了|申请过|不是有吗|申请完了|借过了|有你们的app|下载了)")
register_type("alreadyhave_negative_rule1",r"(没|不|一定要)")
rc_alreadyhave = SingleRuleCreator("keyword_zhongan_alreadyhave") \
        .add_positive_rule("*{alreadyhave_positive_rule11:alreadyhave_positive_rule11}*") \
        .add_negative_rule("*{alreadyhave_negative_rule1:alreadyhave_negative_rule1}*") \
        .add_rule_weight(15)
# 厌恶
register_type("hate_positive_rule11",r"(不用|别|老|不要|再|天天|一直|总是|一天到晚|不停的)")
register_type("hate_positive_rule12",r"(电话)")
register_type("hate_negative_rule1",r"(我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼|的时候|需要时候|晚会|再见|接电话)")
rc_hate = SingleRuleCreator("keyword_zhongan_hate") \
        .add_positive_rule("*{hate_positive_rule11:hate_positive_rule11}*?{hate_positive_rule12:hate_positive_rule12}*") \
        .add_negative_rule("*{hate_negative_rule1:hate_negative_rule1}*") \
        .add_rule_weight(25)
# 需要哪些材料
register_type("material_positive_rule11",r"(贷款|小贷|众安|借款|准备)")
register_type("material_positive_rule12",r"(材料|条件|抵押|手续|什么)")
register_type("material_negative_rule1",r"(银行)")
rc_material = SingleRuleCreator("keyword_zhongan_material") \
        .add_positive_rule("*{material_positive_rule11:material_positive_rule11}*?{material_positive_rule12:material_positive_rule12}*") \
        .add_negative_rule("*{material_negative_rule1:material_negative_rule1}*") \
        .add_rule_weight(15)
register_type("material_positive_rule21",r"(怎么|搁哪|去哪)")
register_type("material_positive_rule22",r"(申请|办理|操作|弄)")
register_type("material_negative_rule2",r"(通过|不了)")
rc_material1 = SingleRuleCreator("keyword_zhongan_material1") \
        .add_positive_rule("*{material_positive_rule21:material_positive_rule21}*?{material_positive_rule22:material_positive_rule22}*") \
        .add_negative_rule("*{material_negative_rule2:material_negative_rule2}*") \
        .add_rule_weight(20)
# 问身份
register_type("identity_positive_rule11",r"(你是|你)")
register_type("identity_positive_rule12",r"(是谁|哪个|人工|叫啥|人|哪位|机器人|姓名)")
register_type("identity_negative_rule1",r"(平台|公司|银行|APP|单位|app|说|城市|地方|贷|打错|骗)")
rc_identity = SingleRuleCreator("keyword_zhongan_identity") \
        .add_positive_rule("*{identity_positive_rule11:identity_positive_rule11}*?{identity_positive_rule12:identity_positive_rule12}*") \
        .add_negative_rule("*{identity_negative_rule1:identity_negative_rule1}*") \
        .add_rule_weight(15)
# 问链接
register_type("link_positive_rule11",r"(链接|连接)")
register_type("link_positive_rule12",r"(什么|哪个)")
register_type("link_negative_rule1",r"(不知道)")
rc_link = SingleRuleCreator("keyword_zhongan_link") \
        .add_positive_rule("*{link_positive_rule11:link_positive_rule11}*?{link_positive_rule12:link_positive_rule12}*") \
        .add_negative_rule("*{link_negative_rule1:link_negative_rule1}*") \
        .add_rule_weight(15)
# 问公司
register_type("company_positive_rule11",r"(什么)")
register_type("company_positive_rule12",r"(平台|公司|单位|APP|众安)")
rc_company = SingleRuleCreator("keyword_zhongan_company") \
        .add_positive_rule("*{company_positive_rule11:company_positive_rule11}*?{company_positive_rule12:company_positive_rule12}*") \
        .add_rule_weight(15)
register_type("company_positive_rule21",r"(不清楚|不明白|不知道|不了解|搞不懂|啥)")
register_type("company_positive_rule22",r"(哪的|哪里|平台|公司|单位|APP|干啥的|干什么|app|哪方面)")
rc_company1 = SingleRuleCreator("keyword_zhongan_company1") \
        .add_positive_rule("*{company_positive_rule21:company_positive_rule21}*?{company_positive_rule22:company_positive_rule22}*") \
        .add_rule_weight(15)
register_type("company_positive_rule31",r"(什么|哪个)")
register_type("company_positive_rule32",r"(平台|公司|单位|APP|业务|网|软件|贷款)")
register_type("company_negative_rule3",r"(没听清|你|不可靠)")
rc_company2 = SingleRuleCreator("keyword_zhongan_company2") \
        .add_positive_rule("*{company_positive_rule31:company_positive_rule31}*?{company_positive_rule32:company_positive_rule32}*") \
        .add_negative_rule("*{company_negative_rule3:company_negative_rule3}*") \
        .add_rule_weight(15)
register_type("company_positive_rule41",r"(贷款吗|干啥的|干什么的|众安小贷|网贷|众安|小贷|金融|软件)")
register_type("company_negative_rule4",r"(不需要|不想要|没想要|没想用|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|不可能贷|不带|不申请|没弄|不使用|没有需求|没有兴趣|没有申请|没收到|没通过|没有通过|没申请|之前点过|不会过|过不了|没说申请|申请不了|不可拿|链接|正规|什么要求|下载|你|一定要|不弄|有过|申请了|不可靠)")
rc_company3 = SingleRuleCreator("keyword_zhongan_company3") \
        .add_positive_rule("*{company_positive_rule41:company_positive_rule41}*") \
        .add_negative_rule("*{company_negative_rule4:company_negative_rule4}*") \
        .add_rule_weight(15)
# 未听清
register_type("nothear_positive_rule11",r"(贷款吗|干啥的|干什么的|众安小贷|网贷|众安|小贷|金融|软件)")
register_type("nothear_negative_rule1",r"(不需要|不想要|没想要|没想用|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|不可能贷|不带|不申请|没弄|不使用|没有需求|没有兴趣|没有申请|没收到|没通过|没有通过|没申请|之前点过|不会过|过不了|没说申请|申请不了|不可拿|链接|正规|什么要求|下载|你|一定要|不弄|有过|申请了|不可靠)")
rc_nothear = SingleRuleCreator("keyword_zhongan_nothear") \
        .add_positive_rule("*{nothear_positive_rule11:nothear_positive_rule11}*") \
        .add_negative_rule("*{nothear_negative_rule1:nothear_negative_rule1}*") \
        .add_rule_weight(15)
# 未收到
register_type("notreceived_positive_rule11",r"(没有|没)")
register_type("notreceived_positive_rule12",r"(收到|看见|看到|接到|注意看|联系|电话)")
register_type("notreceived_negative_rule1",r"(没时间|没空|在忙|没接|一天到晚)")
rc_notreceived = SingleRuleCreator("keyword_zhongan_notreceived") \
        .add_positive_rule("*{notreceived_positive_rule11:notreceived_positive_rule11}*?{notreceived_positive_rule12:notreceived_positive_rule12}*") \
        .add_negative_rule("*{notreceived_negative_rule1:notreceived_negative_rule1}*") \
        .add_rule_weight(15)
register_type("notreceived_positive_rule21",r"(联系过我|发过短信|发信息|发短信|发什么|发过消息|发的信息|发的短信)")
register_type("notreceived_negative_rule2",r"(报警|链接|你他妈|玩意儿)")
rc_notreceived1 = SingleRuleCreator("keyword_zhongan_notreceived") \
        .add_positive_rule("*{notreceived_positive_rule21:notreceived_positive_rule21}*") \
        .add_negative_rule("*{notreceived_negative_rule2:notreceived_negative_rule2}*") \
        .add_rule_weight(15)
# 同意配合
register_type("cooperate_positive_rule11",r"(加|发|留)")
register_type("cooperate_positive_rule12",r"(微信|短信|链接|联系方式|一下|手机|号码|电话|给我)")
register_type("cooperate_negative_rule1",r"(什么|没|不|别|给我发|忙)")
rc_cooperate = SingleRuleCreator("keyword_zhongan_cooperate") \
        .add_positive_rule("*{cooperate_positive_rule11:cooperate_positive_rule11}*?{cooperate_positive_rule12:cooperate_positive_rule12}*") \
        .add_negative_rule("*{cooperate_negative_rule1:cooperate_negative_rule1}*") \
        .add_rule_weight(15)
# 同行
register_type("peer_positive_rule11",r"(做贷款|做放贷|放贷款|是同行|同一个行业|从事借贷|从事贷款|做这方面|从事这方面|做借款|从事借款)")
register_type("peer_negative_rule1",r"(什么|没|不|别|给我发|忙)")
rc_peer = SingleRuleCreator("keyword_zhongan_peer") \
        .add_positive_rule("*{peer_positive_rule11:peer_positive_rule11}*") \
        .add_negative_rule("*{peer_negative_rule1:peer_negative_rule1}*") \
        .add_rule_weight(10)
# 调戏
register_type("flirt_positive_rule11",r"(亿|50万|100万|百万|200万|400万|500万|一千万|1000万|2000万|五千万)")
register_type("flirt_negative_rule1",r"(欠着|鸡巴)")
rc_flirt = SingleRuleCreator("keyword_zhongan_flirt") \
        .add_positive_rule("*{flirt_positive_rule11:flirt_positive_rule11}*") \
        .add_negative_rule("*{flirt_negative_rule1:flirt_negative_rule1}*") \
        .add_rule_weight(25)
register_type("flirt_positive_rule21",r"(不要还|不用还|先用着)")
rc_flirt1 = SingleRuleCreator("keyword_zhongan_flirt") \
        .add_positive_rule("*{flirt_positive_rule21:flirt_positive_rule21}*") \
        .add_rule_weight(15)
# 普通忙
register_type("commonbusy_positive_rule11",r"(现在忙|正忙|不方便|不太方便|上班|开会|没时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|在外面)")
register_type("commonbusy_negative_rule1",r"(开车|开着车|不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|等他|等她|有事吗|打错了|找错人|打错号码|号码错误|不是本人|打错电话|转达|转告|让他)")
rc_commonbusy = SingleRuleCreator("keyword_zhongan_commonbusy") \
        .add_positive_rule("*{commonbusy_positive_rule11:commonbusy_positive_rule11}*") \
        .add_negative_rule("*{commonbusy_negative_rule1:commonbusy_negative_rule1}*") \
        .add_rule_weight(25)
# 年龄不合适
register_type("age_positive_rule11",r"(七八十岁|六七十岁)")
rc_age = SingleRuleCreator("keyword_zhongan_age") \
        .add_positive_rule("*{age_positive_rule11:age_positive_rule11}*") \
        .add_rule_weight(20)
register_type("age_positive_rule21",r"(年龄|年纪|岁数)")
register_type("age_positive_rule22",r"(太大|太小|大)")
rc_age1 = SingleRuleCreator("keyword_zhongan_age") \
        .add_positive_rule("*{age_positive_rule21:age_positive_rule21}*?{age_positive_rule22:age_positive_rule22}*") \
        .add_rule_weight(20)
# 没有钱
register_type("nomoney_positive_rule11",r"(没有钱|没钱还|还不起|还不上|没有钱还|还不上|怕欠钱|没钱|还不了)")
rc_nomoney = SingleRuleCreator("keyword_zhongan_nomoney") \
        .add_positive_rule("*{nomoney_positive_rule11:nomoney_positive_rule11}*") \
        .add_rule_weight(12)
# 利息
register_type("interest_positive_rule11",r"(利息|利率|手续费|厘|免息|高利贷)")
register_type("interest_positive_rule12",r"(不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|100万|200万|400万|500万|一千万|1000万|2000万|五千万|百万)")
rc_interest = SingleRuleCreator("keyword_zhongan_interest") \
        .add_positive_rule("*{interest_positive_rule11:interest_positive_rule11}*?{interest_positive_rule12:interest_positive_rule12}*") \
        .add_rule_weight(20)
# 口碑差
register_type("reputation_positive_rule11",r"(不靠谱|口碑差|骗人|诈骗|骗子|骗我|欺骗|骗来骗去|骗来|诈骗)")
register_type("reputation_negative_rule1",r"(靠不)")
rc_reputation = SingleRuleCreator("keyword_zhongan_reputation") \
        .add_positive_rule("*{reputation_positive_rule11:reputation_positive_rule11}*") \
        .add_negative_rule("*{reputation_negative_rule1:reputation_negative_rule1}*") \
        .add_rule_weight(15)
# 肯定
register_type("sure_positive_rule11",r"(好|可以|行|照|管|好的|办理|办一个|办一下|想要|想用)")
register_type("sure_negative_rule1",r"(不|没|你好|您好|再说|再联系|再看|那么好|可以借|可以贷|可以吗|可以批|行吗|好吗|好多|没空|发|加|微信|链接|短信|银行|哪一行|什么机构|需要抵押|资料|好像|再看|再讲|看一下|看看|没时间|再会|再联系|联系|有时间|事吗|什么程序|忙着|什么条件|给你打电话|没需要|就去申请|在上班|快一点|额度|利息|到时候|再弄|一直忙|再聊吧|知道了|以后|怎么办理|我有|申请一下|你妹|证件|能下来多少|再去接|先忙|给电话|有了|做什么的|现在没时间|好吧|记录|知道|手续|需要多少|开车|考虑|行业|亿|注销|话费|太少|提前还|申请多少|好久|做好|这么好|好不好|需要的时候|需要用的时候|要用的时候|等我需要|开会|没有兴趣|等一下看|看情况吧|回去再申请|忙|你妈的|理财机构|知道了|没收到|需要|想要|转达|同行|贷多少|不是说|挂掉|什么意思|证明|车辆|忘了|几天|五千万|等我需要|几年级|分期|中行|500万|回去弄|我操|操你|妈的|死|滚蛋|你妈|有病|他妈|鸡|神经病|妈个逼)")
rc_sure = SingleRuleCreator("keyword_zhongan_sure") \
        .add_positive_rule("*{sure_positive_rule11:sure_positive_rule11}*") \
        .add_negative_rule("*{sure_negative_rule1:sure_negative_rule1}*") \
        .add_rule_weight(10)
register_type("sure_positive_rule21",r"(没|没有|没得)")
register_type("sure_positive_rule22",r"(问题|什么问题|任何问题)")
register_type("sure_negative_rule2",r"(你好|您好|再说|再联系|再看|那么好|可以借|可以贷|可以吗|可以批|行吗|好吗|好多|没空|发|加|微信|链接|短信|银行|哪一行|什么机构|需要抵押|资料|好像|再看|再讲|看一下|看看|没时间|再会|再联系|联系|有时间|事吗|什么程序|忙着|什么条件|给你打电话|没需要|就去申请|在上班|快一点|额度|利息|到时候|再弄|一直忙|再聊吧|知道了|以后|怎么办理|我有|申请一下|你妹|证件|能下来多少|再去接|先忙|给电话|有了|做什么的|现在没时间|好吧|记录|知道|手续|需要多少|开车|考虑|行业|亿|注销|话费|太少|提前还|申请多少|好久|做好|这么好|好不好|需要的时候|需要用的时候|要用的时候|等我需要|不用|需要|不还|不想|忙|不申请|回去再申请|不缺钱|好申请吗|用不到|不是说|忘了)")
rc_sure1 = SingleRuleCreator("keyword_zhongan_sure1") \
        .add_positive_rule("*{sure_positive_rule21:sure_positive_rule21}*?{sure_positive_rule22:sure_positive_rule22}") \
        .add_negative_rule("*{sure_negative_rule2:sure_negative_rule2}*") \
        .add_rule_weight(12)
register_type("sure_positive_rule31",r"(嗯|哦|对)")
register_type("sure_negative_rule3",r"(不|我)")
rc_sure2 = SingleRuleCreator("keyword_zhongan_sure2") \
        .add_positive_rule("*{sure_positive_rule31:sure_positive_rule31}*") \
        .add_negative_rule("*{sure_negative_rule3:sure_negative_rule3}*") \
        .add_rule_weight(1)
register_type("sure_positive_rule41",r"(是|有)")
rc_sure3 = SingleRuleCreator("keyword_zhongan_sure3") \
        .add_positive_rule("*{sure_positive_rule41:sure_positive_rule41}*") \
        .add_rule_weight(1)
# 开车忙
register_type("drive_positive_rule11",r"(在开车|在高速上|开着车|在车上|开车呢|开车|正开车|开着车)")
register_type("drive_negative_rule1",r"(再说|再讲|不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请)")
rc_drive = SingleRuleCreator("keyword_zhongan_drive") \
        .add_positive_rule("*{drive_positive_rule11:drive_positive_rule11}*") \
        .add_negative_rule("*{drive_negative_rule1:drive_negative_rule1}*") \
        .add_rule_weight(25)
# 黑户
register_type("black_positive_rule11",r"(黑户|黑名单)")
rc_black = SingleRuleCreator("keyword_zhongan_black") \
        .add_positive_rule("*{black_positive_rule11:black_positive_rule11}*") \
        .add_rule_weight(15)
# 还款方式
register_type("repay_positive_rule11",r"(还款方式|怎么还钱|自动还款|怎么还|手动还款|手动还|自动还|提前还)")
rc_repay = SingleRuleCreator("keyword_zhongan_repay") \
        .add_positive_rule("*{repay_positive_rule11:repay_positive_rule11}*") \
        .add_rule_weight(20)
# 额度
register_type("quota_positive_rule11",r"(额度|金额|20万)")
register_type("quota_positive_rule12",r"(太小|小|太少|少|不够|太低|低了)")
register_type("quota_negative_rule1",r"(亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万)")
rc_quota = SingleRuleCreator("keyword_zhongan_quota") \
        .add_positive_rule("*{quota_positive_rule11:quota_positive_rule11}*?{quota_positive_rule12:quota_positive_rule12}") \
        .add_negative_rule("*{quota_negative_rule1:quota_negative_rule1}*") \
        .add_rule_weight(25)
register_type("quota_positive_rule21",r"(能借|能贷款|能贷|能给|能拿|能有|额度|可以借|能借|可以贷|批|备用金|里面|给我)")
register_type("quota_positive_rule22",r"(多少钱|多少)")
register_type("quota_negative_rule2",r"(亿)")
rc_quota1 = SingleRuleCreator("keyword_zhongan_quota1") \
        .add_positive_rule("*{quota_positive_rule21:quota_positive_rule21}*?{quota_positive_rule22:quota_positive_rule22}") \
        .add_negative_rule("*{quota_negative_rule2:quota_negative_rule2}*") \
        .add_rule_weight(25)
# 贷款快不快
register_type("loanspeed_positive_rule11",r"(多久|几天|多长时间|几天|多少天|哪一天|哪天)")
register_type("loanspeed_positive_rule12",r"(下款|放款|拿到钱|拿钱)")
rc_loanspeed = SingleRuleCreator("keyword_zhongan_loanspeed") \
        .add_positive_rule("*{loanspeed_positive_rule11:loanspeed_positive_rule11}*?{loanspeed_positive_rule12:loanspeed_positive_rule12}*") \
        .add_rule_weight(15)
# 不需要
register_type("unwanted_positive_rule11",r"(不需要|不想要|没想要|没想用|不是很需要|没得需要|用不到|不考虑|没考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请)")
register_type("unwanted_negative_rule1",r"(开车|开着车|没时间|没空|骗人|再申请|再办|再搞|在上班|不要老是打电话|怎么借|打电话|打扰|骚扰|忽悠|骗钱|骗子|年纪大|年龄大|申请过|不贷给我|不要还|不用还|不需要还|亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万|打来|借给你|打了|不会弄|百万|抵押|申不申|我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼)")
rc_unwanted = SingleRuleCreator("keyword_zhongan_unwanted") \
        .add_positive_rule("*{unwanted_positive_rule11:unwanted_positive_rule11}*") \
        .add_negative_rule("*{unwanted_negative_rule1:unwanted_negative_rule1}*") \
        .add_rule_weight(30)
register_type("unwanted_positive_rule21",r"(没说|没有|没)")
register_type("unwanted_positive_rule22",r"(申请|需要|要贷款|办理|需求)")
register_type("unwanted_negative_rule2",r"(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间|没通过|什么没有|啥子没有|啥玩意)")
rc_unwanted1 = SingleRuleCreator("keyword_zhongan_unwanted1") \
        .add_positive_rule("*{unwanted_positive_rule21:unwanted_positive_rule21}*?{unwanted_positive_rule22:unwanted_positive_rule22}") \
        .add_negative_rule("*{unwanted_negative_rule2:unwanted_negative_rule2}*") \
        .add_rule_weight(10)
register_type("unwanted_positive_rule31",r"(申请|需要|要贷款|办理)")
register_type("unwanted_positive_rule32",r"(干嘛|干什么)")
register_type("unwanted_negative_rule3",r"(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间)")
rc_unwanted2 = SingleRuleCreator("keyword_zhongan_unwanted2") \
        .add_positive_rule("*{unwanted_positive_rule31:unwanted_positive_rule31}*?{unwanted_positive_rule32:unwanted_positive_rule32}") \
        .add_negative_rule("*{unwanted_negative_rule3:unwanted_negative_rule3}*") \
        .add_rule_weight(10)
# 不信任
register_type("distrust_positive_rule11",r"(不相信|不信|不会信|不会相信|不可靠)")
register_type("distrust_negative_rule1",r"(骗|口碑差)")
rc_distrust = SingleRuleCreator("keyword_zhongan_distrust") \
        .add_positive_rule("*{distrust_positive_rule11:distrust_positive_rule11}*") \
        .add_negative_rule("*{distrust_negative_rule1:distrust_negative_rule1}*") \
        .add_rule_weight(10)
# 不是本人
register_type("noparty_positive_rule11",r"(打错了|打错人|找错人|打错号码|号码错误|不是本人|打错电话|转达|转告|等他|让他|打错手机)")
rc_noparty = SingleRuleCreator("keyword_zhongan_noparty") \
        .add_positive_rule("*{noparty_positive_rule11:noparty_positive_rule11}*") \
        .add_rule_weight(25)
ip = SingleIntentParse()
ip.add_intent("质疑号码信息泄露",rc_disclosure)
ip.add_intent("征信低",rc_lowcredit)
ip.add_intent("征信低",rc_lowcredit1)
ip.add_intent("征信低",rc_lowcredit2)
ip.add_intent("征信低",rc_lowcredit3)
ip.add_intent("脏话骂人",rc_dirty)
ip.add_intent("以后再说",rc_talklater)
ip.add_intent("以后再说",rc_talklater1)
ip.add_intent("以后再说",rc_talklater2)
ip.add_intent("已经有了",rc_alreadyhave)
ip.add_intent("厌恶",rc_hate)
ip.add_intent("需要哪些材料",rc_material)
ip.add_intent("需要哪些材料",rc_material1)
ip.add_intent("问身份",rc_identity)
ip.add_intent("问链接",rc_link)
ip.add_intent("问公司",rc_company)
ip.add_intent("问公司",rc_company1)
ip.add_intent("问公司",rc_company2)
ip.add_intent("问公司",rc_company3)
ip.add_intent("未听清",rc_nothear)
ip.add_intent("未收到",rc_notreceived)
ip.add_intent("未收到",rc_notreceived1)
ip.add_intent("同意配合",rc_cooperate)
ip.add_intent("同行",rc_peer)
ip.add_intent("调戏",rc_flirt)
ip.add_intent("调戏",rc_flirt1)
ip.add_intent("普通忙",rc_commonbusy)
ip.add_intent("年龄不合适",rc_age)
ip.add_intent("年龄不合适",rc_age1)
ip.add_intent("没有钱",rc_nomoney)
ip.add_intent("利息",rc_interest)
ip.add_intent("口碑差",rc_reputation)
ip.add_intent("肯定",rc_sure)
ip.add_intent("肯定",rc_sure1)
ip.add_intent("肯定",rc_sure2)
ip.add_intent("肯定",rc_sure3)
ip.add_intent("开车忙",rc_drive)
ip.add_intent("黑户",rc_black)
ip.add_intent("还款方式",rc_repay)
ip.add_intent("额度",rc_quota)
ip.add_intent("额度",rc_quota1)
ip.add_intent("贷款快不快",rc_loanspeed)
ip.add_intent("不需要",rc_unwanted)
ip.add_intent("不需要",rc_unwanted1)
ip.add_intent("不需要",rc_unwanted2)
ip.add_intent("不信任",rc_distrust)
ip.add_intent("不是本人",rc_noparty)

if __name__ == '__main__':
    res_dict = ip.calculate_intent("你们多久可以下款")
    print(res_dict)
    # print(max(res_dict,key=lambda x:x[1]))
    # res = max(res_dict,key = res_dict.get)
#     res = max(res_dict,key = lambda k:res_dict[k])
#     print("res_dict:{},res:{},res_value:{}".format(res_dict,res,res_dict.get(res)))