from flask import Flask, request, jsonify
from Robot_Keyword_Credit import rule_engine
import json
app=Flask(__name__) #创建新的开始

rengine = rule_engine("/home/cjzhu3/cjzhu3_nlu/rule.bin")

# 更新规则接口
@app.route('/rule/update',methods=['GET','POST'])
def update():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        res = parse_rdg_interface(json_data)
        print("req:{},res:{}".format(req,res))
        rengine.rules_update(res)
        return res

# 规则查询接口
@app.route('/rule/search',methods=['GET','POST'])
def search():
    if request.method == 'POST':
        req = request.get_data(as_text=True)
        json_data = json.loads(req)
        query = json_data['query']
        parent_node = json_data['org']
        res = rengine.rule_parse(parent_node,query)
        print("req:{},res:{}".format(req,res))
        return res.to_json()

# 解析研究院请求body
def parse_rdg_interface(json_data):
        parent_dict = {}
        for key,value in json_data['org'].items():
            parent_node_name = key #父节点名
            parent_node_resources = value['resources']
            son_node_rule = parent_node_resources['rule'] #子节点资源
            son_dict = {}
            for son_node_json in son_node_rule:
                son_node_name = son_node_json['kp']
                son_node_rules = son_node_json['rule']
                rule_arr = []
                for rule in son_node_rules:
                    if rule is not None:
                        tmp_rule = rule.split("/")
                        rule_arr.append({'rule':tmp_rule[0],'weight':tmp_rule[1]})
                son_dict[son_node_name] = rule_arr
            parent_dict[parent_node_name] = son_dict
        return parent_dict


if __name__ == '__main__':
   app.config['JSON_AS_ASCII'] = False
   app.run(host='0.0.0.0',port=1472,debug=True) # 运行开始

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org": {"开场白_yB7b2E": {"use_model_for_token": 0,"use_protocol": 0,"rule_thres": 0.1,"resources": {"synonym": [],"protocol": [],"high_weight": [],"rule": [{"kp": "口碑差_7a8Ax7","rule": [],"priority": []},{"kp": "没小孩-幼儿3-8岁_EAdQu1","rule": [],"priority": []},{"kp": "姓名_oRqvs2","rule": [    "(汤建飞)/1/",    "(汤建飞|建飞汤)/2/"],"priority": [    1,    1]}],"slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1}],"same_class": [],"token": []},"use_rule": 1,"global_token_module": 0,"use_rule_for_token": 0}}}' http://172.21.191.94:1472/rule/update

#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org": {"开场白_yB7b2E": {"use_model_for_token": 0,"use_protocol": 0,"rule_thres": 0.1,"resources": {"synonym": [],"protocol": [],"high_weight": [],"rule": [{"kp": "口碑差_7a8Ax7","rule": ["(口碑差|骗人|诈骗|骗子|骗我|欺骗|骗来骗去|骗来|诈骗)&~(靠不)/15"],"priority": []},{"kp": "不需要_EAdQu1","rule": ["(申请|需要|要贷款|办理)&(干嘛|干什么)&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间)/25","(没说|没有|没)&(申请|需要|要贷款|办理|需求)&~(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间|没通过|什么没有|啥子没有|啥玩意)/25","(不需要|不想要|没想要|没想用|不是很需要|不太需要|没得需要|没有需要|用不到|不考虑|没考虑|不贷款|不想贷|不想办|不想用|不感兴趣|没兴趣|不办|不做|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不太用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请|不准备办|不太想办)&~(骗人|不要老是打电话|怎么借|打电话|打扰|骚扰|忽悠|骗钱|骗子|年纪大|年龄大|申请过|不贷给我|不要还|不用还|不需要还|亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万|打来|借给你|打了|不会弄|百万|抵押|申不申|我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼|不要说)/30"],"priority": []},{"kp": "黑户_oRqvs2","rule": ["(黑户|黑名单)/15"],"priority": []}],"slot_map": [{"kp": "没小孩-幼儿3-8岁_EAdQu1","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1},{"kp": "姓名_oRqvs2","slot_name": "SURNAME","slot_words": [],"slot_model_label": "SURNAME","if_use_word": 0,"if_use_model": 1,"if_use_rule": 1}],"same_class": [],"token": []},"use_rule": 1,"global_token_module": 0,"use_rule_for_token": 0}}}' http://172.21.191.94:1472/rule/update


#curl -i -X POST -H "'Content-type':'application/x-www-form-urlencoded', 'charset':'utf-8', 'Accept': 'text/plain'" -d '{"org":"开场白_yB7b2E","query": "我是黑户"}' http://172.21.191.94:1472/rule/search