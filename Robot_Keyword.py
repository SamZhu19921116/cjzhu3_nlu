#coding=utf-8
import pandas as pd

#######################################################内部/众安小贷/使用的/关键词######################################################
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

####################################################研究院/标注/意向/关键词/规则模板#####################################################
# 质疑号码信息泄露
keyword_zhongan_disclosure = "((怎么|哪里|从哪|为啥|哪个|什么渠道|哪个渠道)&(知道|搞到|弄到|得到|晓得|拿到|找到)&(号码|电话|联系方式))"
# 征信低
keyword_zhongan_lowcredit = "(申请失败|审核失败|没通过|申请不到|通不过|申请不了|审核不通过|信用烂了|征信烂了|征信差劲|逾期|没有资格|不通过|申请不下来|贷不下来|批不下来|没有通过|没通过|不能通过|信用不行|征信不行)&~(骗)"
keyword_zhongan_lowcredit1 = "((征信|信用度|信用卡)&(不好|不高|不行|逾期|问题))"
keyword_zhongan_lowcredit2 = "((申请|借|贷|试了|试过|征信|通过|信誉|诚信|信用)&(不下来|下不来|没额度|不过|不成|不批|不出来|出不来|不成功|不良|不出钱|过不去|不了|不好|不太好|不够))&~(年龄|年纪|岁数|还不了|骗)"
keyword_zhongan_lowcredit3 = "(不贷给我|不好贷|没有额度)&~(骗|已经申请)"
# 脏话骂人
keyword_zhongan_dirty = "(我操|妈的|死|滚蛋|你妈|你妹|有病|他妈|鸡巴|神经病|妈个逼|脑子有问题|有毒|傻逼)&~(操作)"
# 以后再说
keyword_zhongan_talklater = "(看一下|再说|看看|再看|再联系|再会|稍后再打|再试试|看情况|以后再说)&~(开车|发|一遍|一次|额度|忙|有事情|不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|现在忙|正忙|不方便|不太方便|上班|开会|没时间|没啥时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|不需|多少钱|利息)"
keyword_zhongan_talklater1 = "((等|到时候|到时|的时候|以后|后面|过俩天|过几天|下个月|晚上|回去|明天|晚点)&(需要|有空|一下|一会|看|再弄|再用|再办|去办|再试试|试|打|商量|回复|申请))&~(不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|现在忙|正忙|不方便|不太方便|上班|开会|没时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|开车|不需|多少钱|在外面|他)"
keyword_zhongan_talklater2 = "((需要)&(再用|再办|再弄|再注册|再找你|再联系|再来|再登录))&~(不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|现在忙|正忙|不方便|不太方便|上班|开会|没时间|没啥时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|不需|多少钱)"
# 已经有了
keyword_zhongan_alreadyhave = "(申请了|申请过|不是有吗|申请完了|借过了|有你们的app|下载了)&~(没|不|一定要)"
# 厌恶
keyword_zhongan_hate = "((不用|别|老|不要|再|天天|一直|总是|一天到晚|不停的)&(电话))&~(我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼|的时候|需要时候|晚会|再见|接电话)"
# 需要哪些材料
keyword_zhongan_material = "((贷款|小贷|众安|借款|准备)&(材料|条件|抵押|手续|什么))&~(银行)"
keyword_zhongan_material1 = "((怎么|搁哪|去哪)&(申请|办理|操作|弄))&~(通过|不了)"
# 问身份
keyword_zhongan_identity = "((你是|你)&(是谁|哪个|人工|叫啥|人|哪位|机器人|姓名))&~(平台|公司|银行|APP|单位|app|说|城市|地方|贷|打错|骗)"
# 问链接
keyword_zhongan_link = "((链接|连接)&(什么|哪个))&~(不知道)"
# 问公司
keyword_zhongan_company = "(什么)&(平台|公司|单位|APP|众安)"
keyword_zhongan_company1 = "(不清楚|不明白|不知道|不了解|搞不懂|啥)&(哪的|哪里|平台|公司|单位|APP|干啥的|干什么|app|哪方面)"
keyword_zhongan_company2 = "((什么|哪个)&(平台|公司|单位|APP|业务|网|软件|贷款))&~(没听清|你|不可靠)"
keyword_zhongan_company3 = "(贷款吗|干啥的|干什么的|众安小贷|网贷|众安|小贷|金融|软件)&~(不需要|不想要|没想要|没想用|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|不可能贷|不带|不申请|没弄|不使用|没有需求|没有兴趣|没有申请|没收到|没通过|没有通过|没申请|之前点过|不会过|过不了|没说申请|申请不了|不可拿|链接|正规|什么要求|下载|你|一定要|不弄|有过|申请了|不可靠)"   
# 未听清
keyword_zhongan_nothear = "(没听清|没听明白|你在说什么|你说什么|申请什么|申请啥)&~(链接|不知道|店|不批|哪个银行|怎么申请|我都没有|没有问题|上班|什么公司|我没有|没整上去|贷款|哪里)"
# 未收到
keyword_zhongan_notreceived = "((没有|没)&(收到|看见|看到|接到|注意看|联系|电话))&~(没时间|没空|在忙|没接|一天到晚)"
keyword_zhongan_notreceived1 = "(联系过我|发过短信|发信息|发短信|发什么|发过消息|发的信息|发的短信)&~(报警|链接|你他妈|玩意儿)"
# 同意配合
keyword_zhongan_cooperate = "((加|发|留)&(微信|短信|链接|联系方式|一下|手机|号码|电话|给我))&~(什么|没|不|别|给我发|忙)"
# 同行
keyword_zhongan_peer = "(做贷款|做放贷|放贷款|是同行|同一个行业|从事借贷|从事贷款|做这方面|从事这方面|做借款|从事借款)&~(是吗|是吧)"
# 调戏
keyword_zhongan_flirt = "(亿|50万|100万|百万|200万|400万|500万|一千万|1000万|2000万|五千万)&~(欠着|鸡巴)"
keyword_zhongan_flirt1 = "(不要还|不用还|先用着)"
# 普通忙
keyword_zhongan_commonbusy = "(现在忙|正忙|不方便|不太方便|上班|开会|没时间|没空|没时间|没有时间|在忙|在睡觉|在午休|在午睡|在休息|信号不好|在电梯|信号太差|信号差|信号不好|信号不行|在吃饭|太忙了|挺忙|很忙|忙|现在有事|正在忙|有事|在工作|不得空|刚下班|有点事|忙着|没空|在外边|在外面)&~(开车|开着车|不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|等他|等她|有事吗|打错了|找错人|打错号码|号码错误|不是本人|打错电话|转达|转告|让他)"
# 年龄不合适
keyword_zhongan_age = "(七八十岁|六七十岁)"
keyword_zhongan_age1 = "(年龄|年纪|岁数)&(太大|太小|大)"
# 没有钱
keyword_zhongan_nomoney = "(没有钱|没钱还|还不起|还不上|没有钱还|还不上|怕欠钱|没钱|还不了)"
# 利息
keyword_zhongan_interest = "(利息|利率|手续费|厘|免息|高利贷)&(不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|100万|200万|400万|500万|一千万|1000万|2000万|五千万|百万)"
# 口碑差
keyword_zhongan_reputation = "(不靠谱|口碑差|骗人|诈骗|骗子|骗我|欺骗|骗来骗去|骗来|诈骗)&~(靠不)"
# 肯定
keyword_zhongan_sure = "(好|可以|行|照|管|好的|办理|办一个|办一下|想要|想用)&~(不|没|你好|您好|再说|再联系|再看|那么好|可以借|可以贷|可以吗|可以批|行吗|好吗|好多|没空|发|加|微信|链接|短信|银行|哪一行|什么机构|需要抵押|资料|好像|再看|再讲|看一下|看看|没时间|再会|再联系|联系|有时间|事吗|什么程序|忙着|什么条件|给你打电话|没需要|就去申请|在上班|快一点|额度|利息|到时候|再弄|一直忙|再聊吧|知道了|以后|怎么办理|我有|申请一下|你妹|证件|能下来多少|再去接|先忙|给电话|有了|做什么的|现在没时间|好吧|记录|知道|手续|需要多少|开车|考虑|行业|亿|注销|话费|太少|提前还|申请多少|好久|做好|这么好|好不好|需要的时候|需要用的时候|要用的时候|等我需要|开会|没有兴趣|等一下看|看情况吧|回去再申请|忙|你妈的|理财机构|知道了|没收到|需要|想要|转达|同行|贷多少|不是说|挂掉|什么意思|证明|车辆|忘了|几天|五千万|等我需要|几年级|分期|中行|500万|回去弄|我操|操你|妈的|死|滚蛋|你妈|有病|他妈|鸡|神经病|妈个逼)"
keyword_zhongan_sure1 = "((没|没有|没得)&(问题|什么问题|任何问题))&~(你好|您好|再说|再联系|再看|那么好|可以借|可以贷|可以吗|可以批|行吗|好吗|好多|没空|发|加|微信|链接|短信|银行|哪一行|什么机构|需要抵押|资料|好像|再看|再讲|看一下|看看|没时间|再会|再联系|联系|有时间|事吗|什么程序|忙着|什么条件|给你打电话|没需要|就去申请|在上班|快一点|额度|利息|到时候|再弄|一直忙|再聊吧|知道了|以后|怎么办理|我有|申请一下|你妹|证件|能下来多少|再去接|先忙|给电话|有了|做什么的|现在没时间|好吧|记录|知道|手续|需要多少|开车|考虑|行业|亿|注销|话费|太少|提前还|申请多少|好久|做好|这么好|好不好|需要的时候|需要用的时候|要用的时候|等我需要|不用|需要|不还|不想|忙|不申请|回去再申请|不缺钱|好申请吗|用不到|不是说|忘了)"
keyword_zhongan_sure2 = "(嗯|哦|对)&~(不|我)"
keyword_zhongan_sure3 = "(是|有)"
# 开车忙
keyword_zhongan_drive = "(在开车|在高速上|开着车|在车上|开车呢|开车|正开车|开着车)&~(再说|再讲|不需要|不是很需要|没得需要|用不到|不考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请)"
# 黑户
keyword_zhongan_black = "(黑户|黑名单)"
# 还款方式
keyword_zhongan_repay = "(还款方式|怎么还钱|自动还款|怎么还|手动还款|手动还|自动还|提前还)"
# 额度
keyword_zhongan_quota = "((额度|金额|20万)&(太小|小|太少|少|不够|太低|低了))&~(亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万)"
keyword_zhongan_quata1 = "((能借|能贷款|能贷|能给|能拿|能有|额度|可以借|能借|可以贷|批|备用金|里面|给我)&(多少钱|多少))&~(亿)"
# 贷款快不快
keyword_zhongan_loanspeed = "(多久|几天|多长时间|几天|多少天|哪一天|哪天)&(下款|放款|拿到钱|拿钱)"
# 不需要
keyword_zhongan_unwanted = "(不需要|不想要|没想要|没想用|不是很需要|没得需要|用不到|不考虑|没考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请)&~(开车|开着车|没时间|没空|骗人|再申请|再办|再搞|在上班|不要老是打电话|怎么借|打电话|打扰|骚扰|忽悠|骗钱|骗子|年纪大|年龄大|申请过|不贷给我|不要还|不用还|不需要还|亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万|打来|借给你|打了|不会弄|百万|抵押|申不申|我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼)"
keyword_zhongan_unwanted1 = "((没说|没有|没)&(申请|需要|要贷款|办理|需求))&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间|没通过|什么没有|啥子没有|啥玩意)"
keyword_zhongan_unwanted2 = "((申请|需要|要贷款|办理)&(干嘛|干什么))&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间)"
# 不信任
keyword_zhongan_distrust = "(不相信|不信|不会信|不会相信|不可靠)&~(骗|口碑差)"
# 不是本人
keyword_zhongan_noparty = "(打错了|打错人|找错人|打错号码|号码错误|不是本人|打错电话|转达|转告|等他|让他|打错手机)"
########################################################机器人关键词匹配算法部分#######################################################
# 机器人关键词匹配算法-反转匹配部分
def KeyWordMatcherReverse(human_str,regex_str):
    regex_res = []
    regex_tmp = []
    keyword_hit_list = []
    keyword_hitchar_len = 0
    for regex_char in iter(regex_str):
        if regex_char == "(" or regex_char == ")" or regex_char == "|" or regex_char == "&":
            if len(regex_tmp) > 0:
                if "".join(regex_tmp) in human_str:
                    regex_res.append("True")
                    keyword_hitchar_len += len(regex_tmp)
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
            keyword_hitchar_len += len(regex_tmp)
            keyword_hit_list.append("".join(regex_tmp))
        else:
            regex_res.append("False")
    return "".join(regex_res),set(keyword_hit_list),keyword_hitchar_len
# 机器人关键词匹配算法-主函数部分
def KeyWordMatcher(human_str,regex_str):
    split_str = regex_str.split("&~")
    if len(split_str) == 1:
        prefix_matcher_bool_str,prefix_matcher_str_list,prefix_matcher_str_len = KeyWordMatcherReverse(human_str=human_str,regex_str=split_str[0])
    elif len(split_str) == 2:
        prefix_matcher_bool_str,prefix_matcher_str_list,prefix_matcher_str_len = KeyWordMatcherReverse(human_str=human_str,regex_str=split_str[0])
        suffix_matcher_bool_str,suffix_matcher_str_list,suffix_matcher_str_len = KeyWordMatcherReverse(human_str=human_str,regex_str=split_str[1])
    else:
        raise TypeError("Regular Expression Error !")
    
    if len(split_str) == 1:
        return eval(prefix_matcher_bool_str),prefix_matcher_str_list,[],set(human_str),prefix_matcher_str_len
    else:
        return eval(prefix_matcher_bool_str+"&~"+suffix_matcher_bool_str),prefix_matcher_str_list,suffix_matcher_str_list,set(human_str),prefix_matcher_str_len

# 关键词命中节点词典
keyword_dict = {"keyword_zhongan_question":keyword_zhongan_question,"keyword_zhongan_yes":keyword_zhongan_yes,"keyword_zhongan_consider":keyword_zhongan_consider,"keyword_zhongan_no":keyword_zhongan_no,"keyword_zhongan_inconvenient":keyword_zhongan_inconvenient,"keyword_zhongan_complaint":keyword_zhongan_complaint}
# 节点与对应label的映射词典
intent_mapping_dict = {"keyword_zhongan_question":"质疑号码、信息泄露","keyword_zhongan_yes":"肯定","keyword_zhongan_consider":"同意配合","keyword_zhongan_no":"不需要","keyword_zhongan_inconvenient":"普通忙","keyword_zhongan_complaint":"厌恶"}

# 基于RDG关键词
keyword_dict_rdg = {"keyword_zhongan_disclosure":keyword_zhongan_disclosure,"keyword_zhongan_lowcredit":keyword_zhongan_lowcredit,"keyword_zhongan_lowcredit1":keyword_zhongan_lowcredit1,"keyword_zhongan_lowcredit2":keyword_zhongan_lowcredit2,"keyword_zhongan_lowcredit3":keyword_zhongan_lowcredit3,"keyword_zhongan_dirty":keyword_zhongan_dirty,"keyword_zhongan_talklater":keyword_zhongan_talklater,"keyword_zhongan_talklater1":keyword_zhongan_talklater1,"keyword_zhongan_talklater2":keyword_zhongan_talklater2,"keyword_zhongan_alreadyhave":keyword_zhongan_alreadyhave,"keyword_zhongan_hate":keyword_zhongan_hate,"keyword_zhongan_material":keyword_zhongan_material,"keyword_zhongan_material1":keyword_zhongan_material1,"keyword_zhongan_identity":keyword_zhongan_identity,"keyword_zhongan_link":keyword_zhongan_link,"keyword_zhongan_company":keyword_zhongan_company,"keyword_zhongan_company1":keyword_zhongan_company1,"keyword_zhongan_company2":keyword_zhongan_company2,"keyword_zhongan_company3":keyword_zhongan_company3,"keyword_zhongan_nothear":keyword_zhongan_nothear,"keyword_zhongan_notreceived":keyword_zhongan_notreceived,"keyword_zhongan_notreceived1":keyword_zhongan_notreceived1,"keyword_zhongan_cooperate":keyword_zhongan_cooperate,"keyword_zhongan_peer":keyword_zhongan_peer,"keyword_zhongan_flirt":keyword_zhongan_flirt,"keyword_zhongan_flirt1":keyword_zhongan_flirt1,"keyword_zhongan_commonbusy":keyword_zhongan_commonbusy,"keyword_zhongan_age":keyword_zhongan_age,"keyword_zhongan_age1":keyword_zhongan_age1,"keyword_zhongan_nomoney":keyword_zhongan_nomoney,"keyword_zhongan_interest":keyword_zhongan_interest,"keyword_zhongan_reputation":keyword_zhongan_reputation,"keyword_zhongan_sure":keyword_zhongan_sure,"keyword_zhongan_sure1":keyword_zhongan_sure1,"keyword_zhongan_sure2":keyword_zhongan_sure2,"keyword_zhongan_sure3":keyword_zhongan_sure3,"keyword_zhongan_drive":keyword_zhongan_drive,"keyword_zhongan_black":keyword_zhongan_black,"keyword_zhongan_repay":keyword_zhongan_repay,"keyword_zhongan_quota":keyword_zhongan_quota,"keyword_zhongan_quata1":keyword_zhongan_quata1,"keyword_zhongan_loanspeed":keyword_zhongan_loanspeed,"keyword_zhongan_unwanted":keyword_zhongan_unwanted,"keyword_zhongan_unwanted1":keyword_zhongan_unwanted1,"keyword_zhongan_unwanted2":keyword_zhongan_unwanted2,"keyword_zhongan_distrust":keyword_zhongan_distrust,"keyword_zhongan_noparty":keyword_zhongan_noparty}

keyword_dict_tongpei_rdg = {"keyword_zhongan_disclosure":15,"keyword_zhongan_lowcredit":30,"keyword_zhongan_lowcredit1":15,"keyword_zhongan_lowcredit2":20,"keyword_zhongan_lowcredit3":15,"keyword_zhongan_dirty":25,"keyword_zhongan_talklater":15,"keyword_zhongan_talklater1":20,"keyword_zhongan_talklater2":20,"keyword_zhongan_alreadyhave":15,"keyword_zhongan_hate":25,"keyword_zhongan_material":15,"keyword_zhongan_material1":20,"keyword_zhongan_identity":15,"keyword_zhongan_link":15,"keyword_zhongan_company":15,"keyword_zhongan_company1":15,"keyword_zhongan_company2":15,"keyword_zhongan_company3":15,"keyword_zhongan_nothear":15,"keyword_zhongan_notreceived":15,"keyword_zhongan_notreceived1":15,"keyword_zhongan_cooperate":15,"keyword_zhongan_peer":10,"keyword_zhongan_flirt":25,"keyword_zhongan_flirt1":15,"keyword_zhongan_commonbusy":25,"keyword_zhongan_age":20,"keyword_zhongan_age1":20,"keyword_zhongan_nomoney":12,"keyword_zhongan_interest":20,"keyword_zhongan_reputation":15,"keyword_zhongan_sure":10,"keyword_zhongan_sure1":12,"keyword_zhongan_sure2":1,"keyword_zhongan_sure3":1,"keyword_zhongan_drive":25,"keyword_zhongan_black":15,"keyword_zhongan_repay":20,"keyword_zhongan_quota":25,"keyword_zhongan_quata1":25,"keyword_zhongan_loanspeed":15,"keyword_zhongan_unwanted":30,"keyword_zhongan_unwanted1":10,"keyword_zhongan_unwanted2":10,"keyword_zhongan_distrust":10,"keyword_zhongan_noparty":25}

keyword_intent_dict_rdg = {"keyword_zhongan_disclosure":"质疑号码信息泄露","keyword_zhongan_lowcredit":"征信低","keyword_zhongan_lowcredit1":"征信低","keyword_zhongan_lowcredit2":"征信低","keyword_zhongan_lowcredit3":"征信低","keyword_zhongan_dirty":"脏话骂人","keyword_zhongan_talklater":"以后再说","keyword_zhongan_talklater1":"以后再说","keyword_zhongan_talklater2":"以后再说","keyword_zhongan_alreadyhave":"已经有了","keyword_zhongan_hate":"厌恶","keyword_zhongan_material":"需要哪些材料","keyword_zhongan_material1":"需要哪些材料","keyword_zhongan_identity":"问身份","keyword_zhongan_link":"问链接","keyword_zhongan_company":"问公司","keyword_zhongan_company1":"问公司","keyword_zhongan_company2":"问公司","keyword_zhongan_company3":"问公司","keyword_zhongan_nothear":"未听清","keyword_zhongan_notreceived":"未收到","keyword_zhongan_notreceived1":"未收到","keyword_zhongan_cooperate":"同意配合","keyword_zhongan_peer":"同行","keyword_zhongan_flirt":"调戏","keyword_zhongan_flirt1":"调戏","keyword_zhongan_commonbusy":"普通忙","keyword_zhongan_age":"年龄不合适","keyword_zhongan_age1":"年龄不合适","keyword_zhongan_nomoney":"没有钱","keyword_zhongan_interest":"利息","keyword_zhongan_reputation":"口碑差","keyword_zhongan_sure":"肯定","keyword_zhongan_sure1":"肯定","keyword_zhongan_sure2":"肯定","keyword_zhongan_sure3":"肯定","keyword_zhongan_drive":"开车忙","keyword_zhongan_black":"黑户","keyword_zhongan_repay":"还款方式","keyword_zhongan_quota":"额度","keyword_zhongan_quata1":"额度","keyword_zhongan_loanspeed":"贷款快不快","keyword_zhongan_unwanted":"不需要","keyword_zhongan_unwanted1":"不需要","keyword_zhongan_unwanted2":"不需要","keyword_zhongan_distrust":"不信任","keyword_zhongan_noparty":"不是本人"}

def KeyWord_Sort_V1(query):
    import jieba
    # 关键词query对应每个关键词节点命中的情况
    keyword_intent_list = []
    keyword_ishit_list = []
    keyword_hitwordcnt_list = []
    keyword_unhitwordcnt_list = [] 
    for key,value in keyword_dict.items():
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list = KeyWordMatcher(human_str=jieba.lcut(query),regex_str=value)
        keyword_intent_list.append(intent_mapping_dict.get(key))
        keyword_ishit_list.append(matcher_bool)
        keyword_hitwordcnt_list.append(len(prefix_matcher_str_list))
        keyword_unhitwordcnt_list.append(len(suffix_matcher_str_list))
    return pd.DataFrame({"intent":keyword_intent_list,"ishit":keyword_ishit_list,"hitwordcnt":keyword_hitwordcnt_list,"unhitwordcnt":keyword_unhitwordcnt_list})

def KeyWord_Sort_RDG(query):
    import jieba
    # 关键词query对应每个关键词节点命中的情况
    keyword_intent_list = []
    keyword_ishit_list = []
    keyword_hitwordcnt_list = []
    keyword_unhitwordcnt_list = [] 
    for key,value in keyword_dict_rdg.items():
        matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list = KeyWordMatcher(human_str=jieba.lcut(query),regex_str=value)
        keyword_intent_list.append(keyword_intent_dict_rdg.get(key))
        keyword_ishit_list.append(matcher_bool)
        keyword_hitwordcnt_list.append(len(prefix_matcher_str_list))
        keyword_unhitwordcnt_list.append(len(suffix_matcher_str_list))
    return pd.DataFrame({"intent":keyword_intent_list,"ishit":keyword_ishit_list,"hitwordcnt":keyword_hitwordcnt_list,"unhitwordcnt":keyword_unhitwordcnt_list})

if __name__ == '__main__':
    # from RuleNER import RuleNER
    # rules = RuleNER()
    # rules.add_rule(keyword_zhongan_disclosure)
    # # 规则模板方案
    # for ent in rules.extract_entities("你打错了你查下吧我不叫你说的那个人", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)
    #     print("=" * 50)

    # 关键词模板方案
    res = KeyWord_Sort_RDG("你打错人了")
    print(res)

    query = ""
    matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list = KeyWordMatcher(human_str="query",regex_str=keyword_zhongan_disclosure)
    
