#coding=utf-8
import os
import re
import json
import pandas as pd
import pickle
import logging
from flask import current_app

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
logging.getLogger(__name__)
# {'rule':'(黑户|黑名单)','weight':15}
# {父节点：{子意图：[{子意图规则,子意图权重}]}}
# {'开场白_yB7b2E': {'口碑差_7a8Ax7': [], '没小孩-幼儿3-8岁_EAdQu1': [], '姓名_oRqvs2': [{'rule': '(汤建飞)', 'weight': '1'}, {'rule': '(汤建飞|建飞汤)', 'weight': '2'}]}}
class Rule_Engine():
    def __init__(self, rule_model_dir = None):
        self.parent_rules_dict = {}
        if os.path.isdir(rule_model_dir) and len(os.listdir(rule_model_dir)):
            file_name_list = os.listdir(rule_model_dir)
            for file_name in file_name_list:
                if file_name.endswith(".pkl"):
                    self.parent_rules_dict[file_name.replace(".pkl","")] = self.json_deserialization(os.path.join(rule_model_dir,file_name))
                    # logging.info("加载Rule规则模型:{}".format(os.path.join(rule_model_dir,file_name)))
                    # current_app.logger.info("加载Rule规则模型:{}".format(os.path.join(rule_model_dir,file_name)))

    # 将父节点对应下的所有子节点规则信息保存至父节点命名的文件中
    # to_dir/to_file.pkl
    # to_dir/parent_node_name.pkl
    def rules_update(self, from_json, to_dir, to_file):
        if os.path.exists(os.path.join(to_dir, to_file)):
            os.remove(os.path.join(to_dir,to_file))
        self.parent_rules_dict[to_file] = from_json
        self.json_serialize(os.path.join(to_dir,to_file+".pkl"), from_json)
        # logging.info("更新Rule规则模型:{}".format(os.path.join(to_dir,to_file+".pkl")))
        current_app.logger.info("更新Rule规则模型:{}".format(os.path.join(to_dir,to_file+".pkl")))

    # 规则org删除接口
    # org_dir：规则保存目录
    # org_name：org名称，例如：开场白_yB7b2E
    # 删除 org_dir/org_name.pkl
    def rules_delete(self,org_dir,org_name):
        obj_file = os.path.join(org_dir,org_name+".pkl")
        if os.path.exists(obj_file):
            os.remove(obj_file) # 删除规则模型中指定org文件
            self.parent_rules_dict.pop(org_name) # 删除字典中的org值
            current_app.logger.info("删除Rule规则模型:{}".format(obj_file))

    # 通过父亲节点以及query从对应的rule模型中查看匹配结果
    def parse(self, parent_node, query): #parent_node:开场白_yB7b2E
        query = self.clean(query)
        score_max = 0
        matcher_son_rule = ""
        matcher_son_node = ""
        prefix_matcher_list_max = 0
        #{'口碑差_7a8Ax7': [], '没小孩-幼儿3-8岁_EAdQu1': [], '姓名_oRqvs2': [{'rule': '(汤建飞)', 'weight': '1'}, {'rule': '(汤建飞|建飞汤)', 'weight': '2'}]}
        for son_node,rules in self.parent_rules_dict[parent_node].items(): 
            for rule_dict in rules:
                rule = rule_dict['rule']
                weight = rule_dict['weight']
                matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list,prefix_matcher_str_len = self.KeyWordMatcher(human_str = query,regex_str = rule)
                if matcher_bool:
                    score_current = (prefix_matcher_str_len + int(weight)) / len(query)
                else:
                    score_current = (prefix_matcher_str_len) / len(query)
                if prefix_matcher_str_len > prefix_matcher_list_max:
                    if score_current > score_max:
                        score_max = score_current
                        matcher_son_node = son_node
                        matcher_son_rule = rule
        score_max = min(1.0,score_max)
        return pd.DataFrame({"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"},index = [0])

    # 通过父亲节点以及query从对应的rule模型中查看匹配结果
    def rules_parse(self, parent_node, query): #parent_node:开场白_yB7b2E
        query = self.clean(query)
        score_max = 0
        matcher_son_rule = ""
        matcher_son_node = ""
        prefix_matcher_list_max = 0
        #{'口碑差_7a8Ax7': [], '没小孩-幼儿3-8岁_EAdQu1': [], '姓名_oRqvs2': [{'rule': '(汤建飞)', 'weight': '1'}, {'rule': '(汤建飞|建飞汤)', 'weight': '2'}]}
        for son_node,rules in self.parent_rules_dict[parent_node].items(): 
            for rule_dict in rules:
                rule = rule_dict['rule']
                weight = rule_dict['weight']
                matcher_bool,prefix_matcher_str_list,suffix_matcher_str_list,human_str_list,prefix_matcher_str_len = self.KeyWordMatcher(human_str = query,regex_str = rule)
                if matcher_bool:
                    score_current = (prefix_matcher_str_len + int(weight)) / len(query)
                else:
                    score_current = (prefix_matcher_str_len) / len(query)
                if prefix_matcher_str_len > prefix_matcher_list_max:
                    if score_current > score_max:
                        score_max = score_current
                        matcher_son_node = son_node
                        matcher_son_rule = rule
        score_max = min(1.0,score_max)
        return {"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}

    def rules_parse_v0(self, parent_node, query): #parent_node:开场白_yB7b2E
        score_max = 0
        matcher_son_rule = ""
        matcher_son_node = ""
        prefix_matcher_list_max = 0
        #{'口碑差_7a8Ax7': [], '没小孩-幼儿3-8岁_EAdQu1': [], '姓名_oRqvs2': [{'rule': '(汤建飞)', 'weight': '1'}, {'rule': '(汤建飞|建飞汤)', 'weight': '2'}]}
        for son_node,rules in self.parent_rules_dict[parent_node].items(): 
            for rule_dict in rules:
                rule = rule_dict['rule']
                weight = rule_dict['weight']
                matcher_bool,matcher_str_len = self.KeyWordMatcherVer0(human_str = query,regex_str = rule)
                if matcher_bool:
                    score_current = (matcher_str_len + int(weight)) / len(query)
                else:
                    score_current = (matcher_str_len) / len(query)
                if matcher_str_len > prefix_matcher_list_max:
                    if score_current > score_max:
                        score_max = score_current
                        matcher_son_node = son_node
                        matcher_son_rule = rule
        score_max = min(1.0,score_max)
        return {"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}

    def rules_parse_v1(self, parent_node, query, filter_score = 0.7): #parent_node:开场白_yB7b2E
        score_max = 0
        matcher_son_rule = ""
        matcher_son_node = ""
        prefix_matcher_list_max = 0
        #{'口碑差_7a8Ax7': [], '没小孩-幼儿3-8岁_EAdQu1': [], '姓名_oRqvs2': [{'rule': '(汤建飞)', 'weight': '1'}, {'rule': '(汤建飞|建飞汤)', 'weight': '2'}]}
        for son_node,rules in self.parent_rules_dict[parent_node].items(): 
            for rule_dict in rules:
                rule = rule_dict['rule']
                weight = rule_dict['weight']
                if weight == 0: # 当权重为0时,为关键词精准匹配模式
                    matcher_bool,matcher_str_len = self.ExactMatchVer1(human_str = query,regex_str = rule)
                    if matcher_bool:
                        return {"match_score":1.0,"match_node":son_node,"match_rule":rule,"match_module":"keyword"}
                else: # 当权重不为0时，为关键词的算分匹配
                    matcher_bool,matcher_str_len = self.KeyWordMatcherVer0(human_str = query,regex_str = rule)
                    if matcher_bool:
                        score_current = (matcher_str_len + int(weight)) / len(query)
                    else:
                        score_current = (matcher_str_len) / len(query)
                    if matcher_str_len > prefix_matcher_list_max:
                        if score_current > score_max:
                            score_max = score_current
                            matcher_son_node = son_node
                            matcher_son_rule = rule
        score_max = min(1.0,score_max)
        if score_max >= filter_score:
            # current_app.logger.info("请求节点:{},请求文本:{},响应结果:{}".format(parent_node,query,{"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}))
            return {"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}
        else:
            # current_app.logger.info("请求节点:{},请求文本:{},响应结果:{}".format(parent_node,query,{})) 
            return {}
    
    # 规则合法性检查
    def rules_check_v1(self, rules_dict):
        rules = rules_dict['rules']
        status = []
        for rule in rules:
            rule_arr = rule.split("/")
            if len(rule_arr) != 2:
                status.append(False)
                continue
            if self.KeyWordCheckVer0(rule_arr[0]):
                status.append(True)
            else:
                status.append(False)
        return status,rules

    def rules_parse_v2(self, parent_node, query, rule_score=0.7): #parent_node:开场白_yB7b2E
        # current_app.logger.info("输入规则父节点：{}，输入规则文本：{}".format(parent_node,query))
        score_max = 0
        matcher_son_rule = ""
        matcher_son_node = ""
        prefix_matcher_list_max = 0
        #{'口碑差_7a8Ax7': [], '没小孩-幼儿3-8岁_EAdQu1': [], '姓名_oRqvs2': [{'rule': '(汤建飞)', 'weight': '1'}, {'rule': '(汤建飞|建飞汤)', 'weight': '2'}]}
        for son_node,rules in self.parent_rules_dict[parent_node].items(): 
            for rule_dict in rules:
                rule = rule_dict['rule']
                weight = rule_dict['weight']
                if weight == 0: # 当权重为0时,为关键词精准匹配模式
                    matcher_bool,matcher_str_len = self.ExactMatchVer1(human_str = query,regex_str = rule)
                    if matcher_bool:
                        return {"match_score":1.0,"match_node":son_node,"match_rule":rule,"match_module":"keyword"}
                else: # 当权重不为0时，为关键词的算分匹配
                    matcher_bool,matcher_str_len = self.KeyWordMatcherVer0(human_str = query,regex_str = rule)
                    if matcher_bool:
                        score_current = (matcher_str_len + int(weight)) / len(query)
                    else:
                        score_current = (matcher_str_len) / len(query)
                    if matcher_str_len > prefix_matcher_list_max:
                        if score_current > score_max:
                            score_max = score_current
                            matcher_son_node = son_node
                            matcher_son_rule = rule
        score_max = min(1.0,score_max)
        
        # current_app.logger.info("返回结果:{}".format({"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}))
        return {"match_score":score_max,"match_node":matcher_son_node,"match_rule":matcher_son_rule,"match_module":"keyword"}

    # json字符串序列化
    def json_serialize(self,to_file,json_str):
        with open(to_file, 'wb+') as f:
            data = pickle.dumps(json_str)
            f.write(data)
            
    # json字符串反序列化
    def json_deserialization(self,from_file):
        with open(from_file, 'rb+') as f:
            data = pickle.loads(f.read())
            return data

    # 机器人关键词匹配算法-反转匹配部分
    def KeyWordMatcherReverse(self,human_str,regex_str):
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
    def KeyWordMatcher(self,human_str,regex_str):
        split_str = regex_str.split("&~")
        if len(split_str) == 1:
            prefix_matcher_bool_str,prefix_matcher_str_list,prefix_matcher_str_len = self.KeyWordMatcherReverse(human_str=human_str,regex_str=split_str[0])
        elif len(split_str) == 2:
            prefix_matcher_bool_str,prefix_matcher_str_list,prefix_matcher_str_len = self.KeyWordMatcherReverse(human_str=human_str,regex_str=split_str[0])
            suffix_matcher_bool_str,suffix_matcher_str_list,suffix_matcher_str_len = self.KeyWordMatcherReverse(human_str=human_str,regex_str=split_str[1])
        else:
            raise TypeError("Regular Expression Error !")
        
        if len(split_str) == 1:
            return eval(prefix_matcher_bool_str),prefix_matcher_str_list,[],set(human_str),prefix_matcher_str_len
        else:
            return eval(prefix_matcher_bool_str+"&~"+suffix_matcher_bool_str),prefix_matcher_str_list,suffix_matcher_str_list,set(human_str),prefix_matcher_str_len

    # 机器人关键词匹配算法-一步法
    def KeyWordMatcherVer0(self,human_str,regex_str):
        regex_res = []
        regex_tmp = []
        keyword_hit_cnt = 0
        for regex_char in iter(regex_str):
            if regex_char == "(" or regex_char == ")" or regex_char == "|" or regex_char == "&" or regex_char == "~":
                if len(regex_tmp) > 0:
                    if "".join(regex_tmp) in human_str:
                        regex_res.append("True")
                        keyword_hit_cnt += 1
                    else:
                        regex_res.append("False")
                    regex_tmp.clear()
                regex_res.append(regex_char)
            else:
                regex_tmp.append(regex_char)

        if len(regex_tmp) > 0:
            if "".join(regex_tmp) in human_str:
                regex_res.append("True")
                keyword_hit_cnt += 1
            else:
                regex_res.append("False")
        current_app.logger.info("字符串1:{0},字符串2:{1}".format(human_str,regex_str))
        return eval("".join(regex_res)),keyword_hit_cnt

    def ExactMatchVer0(self,human_str,regex_str):
        for word in regex_str.replace("(","").replace(")","").split("|"):
            if word == human_str:
                return True
        return False

    # 机器人关键词规则检查
    def KeyWordCheckVer0(self,regex_str):
        regex_res = []
        regex_tmp = []
        for regex_char in iter(regex_str):
            if regex_char == "(" or regex_char == ")" or regex_char == "|" or regex_char == "&" or regex_char == "~":
                if len(regex_tmp) > 0:
                    regex_res.append("True")
                    regex_tmp.clear()
                regex_res.append(regex_char)
            else:
                regex_tmp.append(regex_char)

        if len(regex_tmp) > 0:
            regex_res.append("True")
        try:
            eval("".join(regex_res))
            return True
        except:
            return False

    # 机器人关键词匹配算法-一步法
    def ExactMatchVer1(human_str,regex_str):
        regex_tmp = []
        for regex_char in iter(regex_str):
            if regex_char == "(" or regex_char == ")" or regex_char == "|" or regex_char == "&" or regex_char == "~":
                if len(regex_tmp) > 0:
                    if "".join(regex_tmp) == human_str:
                        return True, len(regex_tmp)
                    regex_tmp.clear()
            else:
                regex_tmp.append(regex_char)

        if len(regex_tmp) > 0:
            if "".join(regex_tmp) in human_str:
                return True,len(regex_tmp)
        return False, 0