# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import re
import itertools
from collections import namedtuple

WORD_REGEX = r'[\'.\-?!%,;[\]()/\\]|\w+'
def word_tokenize(input_string):
    return [m.group() for m in re.finditer(WORD_REGEX, input_string)]

def get_common_tokens(samples, squash=True):
    common_toks = []
    for s in samples:
        tokens = word_tokenize(s)
        others = [v for v in samples if v != s]
        other_tokens = [word_tokenize(_) for _ in others]
        common_toks.append([t for t in tokens if all(t in toks for toks in other_tokens)])
    if squash:
        return set(flatten(common_toks))
    return common_toks

def get_uncommon_tokens(samples, squash=True):
    uncommon_toks = []
    for s in samples:
        tokens = word_tokenize(s)
        others = [v for v in samples if v != s]
        other_tokens = [word_tokenize(_) for _ in others]
        uncommon_toks.append([t for t in tokens if any(t not in toks for toks in other_tokens)])
    if squash:
        return set(flatten(uncommon_toks))
    return uncommon_toks

def get_exclusive_tokens(samples, squash=True):
    exclusives = []
    for s in samples:
        tokens = word_tokenize(s)
        others = [v for v in samples if v != s]
        other_tokens = flatten([word_tokenize(_) for _ in others])
        exclusives.append([t for t in tokens if t not in other_tokens])
    if squash:
        return set(flatten(exclusives))
    return exclusives

def flatten(some_list, tuples=True):
    _flatten = lambda l: [item for sublist in l for item in sublist]
    if tuples:
        while any(isinstance(x, list) or isinstance(x, tuple) for x in some_list):
            some_list = _flatten(some_list)
    else:
        while any(isinstance(x, list) for x in some_list):
            some_list = _flatten(some_list)
    return some_list

def chunk(text, delimiters, strip=True):
    pattern = f"({'|'.join(list(delimiters))})" # 格式化字符串:f"{变量}" ==> "变量"
    pts = re.split(pattern, text)
    if strip:
        return [p.strip() for p in pts if p.strip()]
    else:
        return pts

def chunk_list(some_list, delimiters):
    return [list(y) for x, y in itertools.groupby(some_list, lambda z: z in delimiters) if not x]

def get_common_chunks(samples, squash=True):
    toks = get_uncommon_tokens(samples)
    chunks = [chunk_list(word_tokenize(s), toks) for s in samples]
    chunks = [[" ".join(_) for _ in s] for s in chunks]
    if squash:
        return set(flatten(chunks))
    return chunks

def get_uncommon_chunks(samples, squash=True):
    toks = get_common_tokens(samples)
    chunks = [chunk_list(word_tokenize(s), toks) for s in samples]
    chunks = [[" ".join(_) for _ in s] for s in chunks]
    if squash:
        return set(flatten(chunks))
    return chunks

def get_exclusive_chunks(samples, squash=True):
    toks = list(get_common_tokens(samples)) + list(get_uncommon_tokens(samples))
    toks = [t for t in toks if t not in get_exclusive_tokens(samples)]
    chunks = [chunk_list(word_tokenize(s), toks) for s in samples]
    chunks = [[" ".join(_) for _ in s] for s in chunks]
    if squash:
        return set(flatten(chunks))
    return chunks

def find_spans(text, samples):
    chunks = chunk(text, samples, strip=False)
    spans = []
    idx = 0
    for sequence in chunks:
        if sequence in samples:
            end = idx + len(sequence)
            spans.append((idx, end, sequence))
        idx += len(sequence)
    return spans

# 取自标准re模块 减去本模块自己要用的"*{}"三个符号
# SPECIAL_CHARS = {i: "\\" + chr(i) for i in b"()[]?+-|^$\\.&~# \t\n\r\v\f"}
SPECIAL_CHARS = {i: "\\" + chr(i) for i in b"&~# \t\n\r\v\f"}

# a regex that ensures all groups to be non-capturing. Otherwise they would appear in the matches
TYPE_CLEANUP_REGEX = re.compile(r"(?<!\\)\((?!\?)") #查找前面不是\,后面不是？的（
# (?<!exp2)exp1:查找前面不是exp2的exp1
# exp1(?!exp2)：查找后面不是exp2的exp1

# `types` is the dict of known types that is filled with register_type
Type = namedtuple("Type", "regex converter")
types = {}

def register_type(name, regex, converter=str):
    """ register a type to be available for the {value:type} matching syntax """
    cleaned = TYPE_CLEANUP_REGEX.sub("(?:", regex) 
    # 将regex中前面不是'\'，后面不是'？'的'('替换成'(?:'
    # (?:exp):匹配exp，不捕获匹配的文本，也不给此组分配组号
    types[name] = Type(regex=cleaned, converter=converter)

# include some useful basic types
register_type("int", r"[+-]?[0-9]+", int)
register_type("float", r"[+-]?([0-9]*[.])?[0-9]+", float)
register_type("letters", r"[a-zA-Z]+")

# found on https://ihateregex.io/
register_type("bitcoin", r"(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}")
register_type("email", r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
register_type("ssn", r"(?!0{3})(?!6{3})[0-8]\d{2}-(?!0{2})\d{2}-(?!0{4})\d{4}")
register_type(
    "ipv4",
    (
        r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]"
        r"?)){3}"
    ),
)
register_type(
    "url",
    (
        r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA"
        r"-Z0-9()!@:%_\+.~#?&\/\/=]*)"
    ),
)

register_type(
    # Visa, MasterCard, American Express, Diners Club, Discover, JCB
    "ccard",
    (
        r"(^4[0-9]{12}(?:[0-9]{3})?$)|(^(?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6]["
        r"0-9]{2}|27[01][0-9]|2720)[0-9]{12}$)|(3[47][0-9]{13})|(^3(?:0[0-5]|[68][0-9])"
        r"[0-9]{11}$)|(^6(?:011|5[0-9]{2})[0-9]{12}$)|(^(?:2131|1800|35\d{3})\d{11}$)"
    ),
)
register_type(
    "ipv6",
    (
        r"(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA"
        r"-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){"
        r"1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3"
        r"}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0"
        r"-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:"
        r"(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5"
        r"]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0"
        r"-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,"
        r"3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))"
    ),
)

## 用户定义规则处理类
class RegexMatcher:
    def __init__(self,pattern="*",case_sensitive=True):
        #self._converters = {}
        self._pattern = pattern
        self._case_sensitive = case_sensitive
        self._regex_compiled = self._create_regex_compiled(pattern)

    def _field_repl(self, matchobj):
        # 命名体有类型声明
        match = re.search(r"\{(\w+):(\w+)\}", matchobj.group(0))
        if match:
            name, type_ = match.groups()
            # print("name:{0} ==> type:{1} ==> {2}".format(name,type_,types[type_].regex))
            # register this field to convert it later
            #self.converters[name] = types[type_].converter
            # print(r"(?P<%s>%s)" % (name, types[type_].regex))
            return r"(?P<%s>%s)" % (name, types[type_].regex)
        # 命名体无类型声明
        match = re.search(r"\{(\w+)\}", matchobj.group(0))
        if match:
            name = match.group(1)
            return r"(?P<%s>.*)" % name

    def _create_regex_compiled(self, pattern):
        flags = 0 if self._case_sensitive else re.IGNORECASE
        # self._converters.clear()  # empty converters
        result = pattern.translate(SPECIAL_CHARS)  # escape special chars
        result = result.replace("*", r".*")  # handle wildcard
        result = re.sub(r"\{\}", r"(.*)", result)  # handle unnamed group
        result = re.sub(r"\{([^\}]*)\}", self._field_repl, result)  # handle named group
        regex_str = r"%s" % result #r"^%s$" % result
        # print("regex:{}".format(regex_str))
        # print("regex_parttern:{}".format(re.compile(regex_str, flags=flags).pattern))
        return re.compile(regex_str, flags=flags)

    def _match(self, string):
        match = self._regex_compiled.match(string)
        if match:
            result = match.groupdict()
            return result, self._regex_compiled.pattern
            # for key, converter in self._converters.items():
            #     result[key] = converter(result[key])
            # return result
        return None,self._regex_compiled.pattern

class Entity:
    def __init__(self, entity_value, entity_name="", source_text="", rule_dict_hit=None, rule_dict_set=None, confidence=1, ignore_case=True):
        self._entity_name = entity_name
        self._entity_value = entity_value
        self._source_text = source_text
        self.ignore_case = ignore_case
        # 命中规则或词典集
        if rule_dict_hit and not isinstance(rule_dict_hit, list) and not isinstance(rule_dict_hit, tuple):
            rule_dict_hit = [rule_dict_hit]
        self._rule_dict_hit = rule_dict_hit or []
        # 全部规则或词典集
        if rule_dict_set and not isinstance(rule_dict_set, list) and not isinstance(rule_dict_set, tuple):
            rule_dict_set = [rule_dict_set]
        self._rule_dict_set = rule_dict_set or []
        self._confidence = confidence

    @property
    def spans(self):
        if self.ignore_case:
            spans = find_spans(self.source_text.lower(), [self.entity_value.lower()])
        else:
            spans = find_spans(self.source_text, [self.entity_value])
        return [(s[0], s[1]) for s in spans]

    @property
    def indexes(self):
        return [i[0] for i in self.spans]

    @property
    def occurrence_number(self):
        return len(self.spans)

    @property
    def confidence(self):
        return self._confidence

    @property
    def rule_dict_hit(self):
        return self._rule_dict_hit

    @property
    def rule_dict_set(self):
        return self._rule_dict_set

    @property
    def entity_name(self):
        return self._entity_name

    @property
    def entity_value(self):
        return self._entity_value

    @property
    def source_text(self):
        return self._source_text

    def as_json(self):
        return {"entity_name": self.entity_name, "entity_value": self.entity_value,"spans": self.spans, "source_text": self.source_text, "confidence": self.confidence, "rule_dict_hit": [r for r in self.rule_dict_hit], "rule_dict_set": [r.as_json() for r in self.rule_dict_set]}

    def __repr__(self):
        return self.entity_name + ":" + self.entity_value

# 字典抽取槽值处理类
class Dictionary:
    def __init__(self, name, dictionary):
        self._name = name
        self._dictionary = dictionary

    @property
    def name(self):
        return self._name

    @property
    def dictionary(self):
        return self._dictionary

    def __repr__(self):
        return self.name

    def as_json(self):
        return {"name": self.name, "dictionary": self._dictionary}

class DictionaryNER:
    def __init__(self):
        self._dictionary = {}

    def is_match(self, text, entity):
        entities = []
        if isinstance(entity, str):
            entities = self._dictionary[entity]
        if isinstance(entity, Entity):
            entities = [entity]
        for ent in entities:
            if re.findall(r'\b' + ent.entity_value.lower() + r"\b", text.lower()):
                return True
        return False

    @property
    def dictionary(self):
        return self._dictionary

    def add_dictionary(self, name, dictionary):
        if isinstance(dictionary, str):
            dictionary = [dictionary]
        if name not in self._dictionary:
            self._dictionary[name] = []
        for dict in dictionary:
            self._dictionary[name].append(Dictionary(name, dict))

    def extract_entities(self, text, as_json=False):
        for name,dict in self._dictionary.items():
            dictionary = flatten([d.dictionary for d in dict])
            for d in dictionary:
                if re.findall(r'\b' + d.lower() + r"\b", text.lower()):
                    if as_json:
                        yield Entity(entity_value=d,entity_name=name,source_text=text,rule_dict_hit=d,rule_dict_set=dict).as_json()
                    else:
                        yield Entity(entity_value=d,entity_name=name,source_text=text,rule_dict_hit=d,rule_dict_set=dict)

    def in_place_annotation(self, text):
        new_text = text
        indexes = {}
        # map ents to indexes
        for ent in self.extract_entities(text):
            indexs = ent.indexes
            for index in indexs:
                index += len(ent.entity_value)
                if index not in indexes:
                    indexes[index] = []
                if (ent.entity_value, ent.entity_name) not in indexes[index]:
                    indexes[index] += [(ent.entity_value, ent.entity_name)]
        # generate annotation
        for index in indexes:
            ano = "(" + "|".join([i[1] for i in indexes[index]]) + ")"
            indexes[index] = ano
        # replace, last index first
        sor = sorted(indexes.keys(), reverse=True)
        for i in sor:
            new_text = new_text[:i] + indexes[i] + new_text[i:]
        return new_text

## 规则处理槽值抽取类
class Rule:
    def __init__(self, name, rules):
        self._name = name
        self._rules = rules

    @property
    def name(self):
        return self._name

    @property
    def rules(self):
        return self._rules

    def __repr__(self):
        return self.name

    def as_json(self):
        return {"name": self.name, "rules": self._rules}

class RuleNER():
    def __init__(self):
        self._rules = {}

    @property
    def rules(self):
        return self._rules

    def add_rule(self, name, rules):
        if isinstance(rules, str):
            rules = [rules]
        if name not in self._rules:
            self._rules[name] = []
        rules = [r.lower() for r in rules]
        self._rules[name].append(Rule(name, rules))

    def extract_entities(self, text, as_json=False):
        for name, rules in self._rules.items():
            regexes = flatten([r.rules for r in rules])
            print("regexes:{}".format(regexes))
            for r in regexes:
                entities,regex_compiled = RegexMatcher(r,case_sensitive=True)._match(text)
                if entities is None:
                    entities,regex_compiled = RegexMatcher(r,case_sensitive=False)._match(text)
                #print("items:{}".format(entities))
                if entities is not None:
                    for k, v in entities.items():
                        if as_json:
                            yield Entity(entity_value=v, entity_name=k, source_text=text,rule_dict_hit=r,rule_dict_set=rules).as_json()
                        else:
                            yield Entity(entity_value=v, entity_name=k, source_text=text,rule_dict_hit=r,rule_dict_set=rules)

class SingleRuleCreator():
    def __init__(self, rule_name):
        self._rule_name = rule_name
        self._positive_rule = None
        self._negative_rule = None
        self._rule_weight = None
    @property
    def positive_rule(self):
        return self._positive_rule
    @property
    def negative_rule(self):
        return self._negative_rule
    @property
    def rule_weight(self):
        return self._rule_weight
    # 添加负向规则
    def add_positive_rule(self,rule):
        if isinstance(rule,str):
            self._positive_rule = rule
        else:
            raise TypeError("Rule Type Error!")
        return self
    # 添加正向规则
    def add_negative_rule(self,rule):
        if isinstance(rule,str):
            self._negative_rule = rule
        else:
            raise TypeError("Rule Type Error!")
        return self
    # 给当前规则添加奖励与惩罚权重
    def add_rule_weight(self,weight):
        if isinstance(weight,int):
            self._rule_weight = weight
        elif weight.isdigit():
            self._rule_weight = int(weight)
        else:
            raise TypeError('Rule Weight Type Error!')
        return self
    #  
    def build(self):
        return {"rule_name": self._rule_name, "positive_rule": self._positive_rule, "negative_rule": self._negative_rule, "rules_weight":self._rule_weight}

class SingleIntentParse():
    def __init__(self):
        self.intents = []

    def add_intent(self, intent_name,rule_creator):
        if isinstance(rule_creator, SingleRuleCreator):
            rule = rule_creator.build()
        self.intents.append((intent_name,rule))

    # 计算负对象是否匹配上
    # (怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话) &~ (手机号有|手机里有|手机有)
    # 该表达式包含两部分：positive_rules:(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话)
    #                   negative_rules:(手机号有|手机里有|手机有)
    # 本函数主要匹配negative_rules部分，若该negative_rules部分匹配上则说明该意向肯定不被命中，否则的话说明该意向可以被命中
    def calculate_negative_value(self, query):
        negative_matcher_flag = False
        negative_matcher_json = {}
        for intent_name,intent in self.intents.items():
            for name,rules in intent['negative_rules'].items():
                regexes = flatten([r.rules for r in rules])
                for r in regexes:
                    entities, regex_compiled = RegexMatcher(r,case_sensitive=True)._match(query)
                    if entities is not None:
                        negative_matcher_flag = True
                        for k, v in entities.items():
                            negative_matcher_json = Entity(entity_value=v, entity_name=k, source_text=query,rule_dict_hit=r,rule_dict_set=rules).as_json()
        if negative_matcher_flag:
            return intent_name,0,negative_matcher_json
        else:
            return intent_name,1,negative_matcher_json

    def calculate_intent(self, query):
        res_intent_tuple = []
        # tuple(intent_name, {"rule_name": self._rule_name, "positive_rule": self._positive_rule, "negative_rule": self._negative_rule, "rules_weight":self._rule_weight})
        for intent_name,rule_creator in self.intents:
            remainder = query
            positive_matcher_len = 0
            # 先判断negative_rules负向规则集是否匹配上，若匹配上则直接跳出本循环(本意向)
            if rule_creator['negative_rule'] is not None:
                entities,regex_compiled = RegexMatcher(rule_creator['negative_rule'],case_sensitive=True)._match(query)
                if entities is not None:
                    res_intent_tuple.append((intent_name,0,"",regex_compiled,entities))
                    continue
            # 然后再判断positive_rules正向规则集是否匹配上，若匹配上
            if rule_creator['positive_rule'] is not None:
                entities,regex_compiled = RegexMatcher(rule_creator['positive_rule'],case_sensitive=True)._match(query)
                if entities is not None:
                    remainder = get_utterance_remainder(remainder, list(entities.values()))
                    for k, v in entities.items():
                        positive_matcher_len += len(v)
                    res_intent_tuple.append((intent_name,int((positive_matcher_len + rule_creator['rules_weight']) / len(query)),regex_compiled,"",entities))
        if len(res_intent_tuple) > 0:
            (max_intent_name,max_intent_score,max_intent_positive_rule,max_intent_negative_rule,entity) = max(res_intent_tuple,key=lambda x:x[1])
        else:
            max_intent_name = ""
            max_intent_score = 0.0
            max_intent_positive_rule = ""
            max_intent_negative_rule = ""
            entity = ""
                    
        return (max_intent_name,min(1.0,max_intent_score),max_intent_positive_rule,max_intent_negative_rule,entity)

class RuleCreator():
    def __init__(self,rule_name):
        self._rule_name = rule_name
        self._positive_rules = {}
        self._negative_rules = {}
        self._rule_weight = {}
    
    @property
    def positive_rules(self):
        return self._positive_rules
    
    @property
    def negative_rules(self):
        return self._negative_rules

    @property
    def rule_weight(self):
        return self._rule_weight

    # 添加规则:(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话) &~ (手机号有|手机里有|手机有)
    # 该表达式包含两部分：positive_rules:(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话)
    #                   negative_rules:(手机号有|手机里有|手机有)
    def add_positive_rule(self, name, rules):
        if isinstance(rules, str):
            rules = [rules]
        if name not in self._positive_rules:
            self._positive_rules[name] = []
        rules = [r.lower() for r in rules]
        self._positive_rules[name].append(Rule(name, rules))
        return self

    # 添加规则:(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话) &~ (手机号有|手机里有|手机有)
    # 该表达式包含两部分：positive_rules:(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话)
    #                   negative_rules:(手机号有|手机里有|手机有)
    def add_negative_rule(self, name, rules):
        if isinstance(rules, str):
            rules = [rules]
        if name not in self._negative_rules:
            self._negative_rules[name] = []
        rules = [r.lower() for r in rules]
        self._negative_rules[name].append(Rule(name, rules))
        return self

    # 给当前规则添加奖励与惩罚权重
    def add_rule_weight(self,weight):
        if isinstance(weight,int):
            self._rule_weight[self._rule_name] = weight
        elif weight.isdigit():
            self._rule_weight[self._rule_name] = int(weight)
        else:
            raise TypeError('Rule Weight Type Error!')
        return self
            
    def build(self):
        return {"rule_name": self._rule_name, "positive_rules": self._positive_rules, "negative_rules": self._negative_rules, "rules_weight":self._rule_weight}

# 获取语料剩余部分的词
def get_utterance_remainder(utterance, samples, as_string=True):
    chunks = flatten([word_tokenize(s) for s in samples])
    words = [t for t in word_tokenize(utterance) if t not in chunks]
    # print("utterance:{},sample:{},chunks:{},words:{}".format(utterance,samples,chunks,words))
    if as_string:
        return " ".join(words)
    return words

class IntentParse():
    def __init__(self):
        self.intents = []

    # build with RuleCreator
    def add_intent(self, intent_name,intent_creator):
        if isinstance(intent_creator, RuleCreator):
            intent = intent_creator.build()
        self.intents.append((intent_name,intent))
        # self.intents[intent_name] = intent

    # 计算负对象是否匹配上
    # (怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话) &~ (手机号有|手机里有|手机有)
    # 该表达式包含两部分：positive_rules:(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话)
    #                   negative_rules:(手机号有|手机里有|手机有)
    # 本函数主要匹配negative_rules部分，若该negative_rules部分匹配上则说明该意向肯定不被命中，否则的话说明该意向可以被命中
    def calculate_negative_value(self, query):
        negative_matcher_flag = False
        negative_matcher_json = {}
        for intent_name,intent in self.intents.items():
            for name,rules in intent['negative_rules'].items():
                regexes = flatten([r.rules for r in rules])
                for r in regexes:
                    entities = RegexMatcher(r,case_sensitive=True)._match(query)
                    if entities is not None:
                        negative_matcher_flag = True
                        for k, v in entities.items():
                            negative_matcher_json = Entity(entity_value=v, entity_name=k, source_text=query,rule_dict_hit=r,rule_dict_set=rules).as_json()
        if negative_matcher_flag:
            return intent_name,0,negative_matcher_json
        else:
            return intent_name,1,negative_matcher_json

    def calculate_intent(self, query):
        intent_score_dict = {}
        for intent_name,intent in self.intents: #.items()
            positive_matcher_flag = False
            remainder = query
            positive_matcher_len = 0
            
            # 先判断negative_rules负向规则集是否匹配上，若匹配上则直接跳出本循环(本意向)
            if len(intent['negative_rules']) != 0:
                for name, rules in intent['negative_rules'].items():
                    regexes = flatten([r.rules for r in rules])
                    for r in regexes:
                        entities = RegexMatcher(r,case_sensitive=True)._match(query)
                        if entities is not None:
                            intent_score_dict[intent_name] = 0.0
                            break
            
            # 然后再判断positive_rules正向规则集是否匹配上，若匹配上
            for name,rules in intent['positive_rules'].items():
                regexes = flatten([r.rules for r in rules])
                for r in regexes:
                    entities = RegexMatcher(r,case_sensitive=True)._match(query)
                    if entities is not None:
                        positive_matcher_flag = True
                        remainder = get_utterance_remainder(remainder, list(entities.values()))
                        for k, v in entities.items():
                            positive_matcher_len += len(v)

            if intent_name not in intent_score_dict.keys():
                intent_score_dict[intent_name] = 0
            # 取最小值
            if positive_matcher_flag:
                intent_score_dict[intent_name] = max(intent_score_dict[intent_name],min(1.0,int((positive_matcher_len + intent['rules_weight'][intent['rule_name']]) / len(query))))
            else:
                intent_score_dict[intent_name] = max(intent_score_dict[intent_name],int((positive_matcher_len) / len(query)))

        return intent_score_dict

if __name__ == "__main__":

    # # 注册规则关键词集 
    # register_type("positive_rule1",r"(不用|别|老|不要|再|天天|一直|总是|一天到晚|不停的)")
    # register_type("positive_rule2",r"(电话)")
    # register_type("negative_rule",r"(我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼|的时候|需要时候|晚会|再见|接电话)")

    # # 规则设计器
    # rc = RuleCreator("zhongan_question") \
    #     .add_positive_rule("positive","*{positive1:positive_rule1}*?{positive2:positive_rule2}") \
    #     .add_negative_rule("negative","*{negative:negative_rule}*") \
    #     .add_rule_weight(25)
    
    # # 意图解析器
    # ip = IntentParse()
    # ip.add_intent("zhongan_question",rc)
    # # print("negative_value:{}".format(ip.calculate_negative_value("你们是怎么知道我的电话")))
    # print("positive_value:{}".format(ip.calculate_intent("别老给我打电话了")))

    # 不需要
    register_type("unwanted_positive_rule11",r"(不需要|不想要|没想要|没想用|不是很需要|没得需要|用不到|不考虑|没考虑|不贷款|不想贷|不想用|不感兴趣|没兴趣|不办|用不上|用不着|用不到|我有钱|不缺钱|不差钱|借到了|不用|不大用|不怎么用|不想借|不申请|不想申请|不借|不要|不贷|不弄|没有贷|不可能贷|不带|不申请|没弄|不用还|不使用|算了|不缺钱|没有需求|没有兴趣|没有贷款需求|为啥要申请|为什么要申请)")
    register_type("unwanted_negative_rule1",r"(开车|开着车|没时间|没空|骗人|再申请|再办|再搞|在上班|不要老是打电话|怎么借|打电话|打扰|骚扰|忽悠|骗钱|骗子|年纪大|年龄大|申请过|不贷给我|不要还|不用还|不需要还|亿|50万|100万|200万|400万|500万|一千万|1000万|2000万|五千万|打来|借给你|打了|不会弄|百万|抵押|申不申|我操|妈的|死|滚蛋|你妈|有病|他妈|鸡巴|神经病|妈个逼)")
    rc_unwanted = SingleRuleCreator("keyword_zhongan_unwanted") \
        .add_positive_rule("*{unwanted_positive_rule11:unwanted_positive_rule11}*") \
        .add_negative_rule("*{unwanted_negative_rule1:unwanted_negative_rule1}*") \
        .add_rule_weight(30)
    # 贷款快不快
    register_type("loanspeed_positive_rule11",r"(多久|几天|多长时间|几天|多少天|哪一天|哪天)")
    register_type("loanspeed_positive_rule12",r"(下款|放款|拿到钱|拿钱)")
    rc_loanspeed = SingleRuleCreator("keyword_zhongan_loanspeed") \
            .add_positive_rule("*{loanspeed_positive_rule11:loanspeed_positive_rule11}*?{loanspeed_positive_rule12:loanspeed_positive_rule12}*") \
            .add_rule_weight(15)

    ip = SingleIntentParse()
    ip.add_intent("不需要",rc_unwanted)
    ip.add_intent("不需要",rc_loanspeed)
    print("positive_value:{}".format(ip.calculate_intent("我真的不需要")))
    
    register_type("surname",r"(赵|钱|孙|李|周|吴|郑|王|冯|陈|褚|卫|蒋|沈|韩|杨|朱|秦|尤|许|何|吕|施|张|孔|曹|严|华|金|魏|陶|姜|戚|谢|邹|喻|柏|水|窦|章|云|苏|潘|葛|奚|范|彭|郎|鲁|韦|昌|马|苗|凤|花|方|俞|任|袁|柳|酆|鲍|史|唐|费|廉|岑|薛|雷|贺|倪|汤|滕|殷|罗|毕|郝|邬|安|常|乐|于|时|傅|皮|卞|齐|康|伍|余|元|卜|顾|孟|平|黄|和|穆|萧|尹|姚|邵|湛|汪|祁|毛|禹|狄|米|贝|明|臧|计|伏|成|戴|谈|宋|茅|庞|熊|纪|舒|屈|项|祝|董|梁|杜|阮|蓝|闵|席|季|麻|强|贾|路|娄|危|江|童|颜|郭|梅|盛|林|刁|钟|徐|邱|骆|高|夏|蔡|田|樊|胡|凌|霍|虞|万|支|柯|昝|管|卢|莫|经|房|裘|缪|干|解|应|宗|丁|宣|贲|邓|郁|单|杭|洪|包|诸|左|石|崔|吉|钮|龚|程|嵇|邢|滑|裴|陆|荣|翁|荀|羊|於|惠|甄|曲|家|封|芮|羿|储|靳|汲|邴|糜|松|井|段|富|巫|乌|焦|巴|弓|牧|隗|山|谷|车|侯|宓|蓬|全|郗|班|仰|秋|仲|伊|宫|宁|仇|栾|暴|甘|钭|厉|戎|祖|武|符|刘|景|詹|束|龙|叶|幸|司|韶|郜|黎|蓟|薄|印|宿|白|怀|蒲|台|从|鄂|索|咸|籍|赖|卓|蔺|屠|蒙|池|乔|阴|欎|胥|能|苍|双|闻|莘|党|翟|谭|贡|劳|逄|姬|申|扶|堵|冉|宰|郦|雍|郤|璩|桑|桂|濮|牛|寿|通|边|扈|燕|冀|郏|浦|尚|农|温|别|庄|晏|柴|瞿|阎|充|慕|连|茹|习|宦|艾|鱼|容|向|古|易|慎|戈|廖|庾|终|暨|居|衡|步|都|耿|满|弘|匡|国|文|寇|广|禄|阙|东|殴|殳|沃|利|蔚|越|夔|隆|师|巩|厍|聂|晁|勾|敖|融|冷|訾|辛|阚|那|简|饶|空|曾|毋|沙|乜|养|鞠|须|丰|巢|关|蒯|相|查|后|荆|红|游|竺|权|逯|盖|益|桓|公|万俟|司马|上官|欧阳|夏侯|诸葛|闻人|东方|赫连|皇甫|尉迟|公羊|澹台|公冶|宗政|濮阳|淳于|单于|太叔|申屠|公孙|仲孙|轩辕|令狐|钟离|宇文|长孙|慕容|鲜于|闾丘|司徒|司空|亓官|司寇|仉|督|子车|颛孙|端木|巫马|公西|漆雕|乐正|壤驷|公良|拓跋|夹谷|宰父|谷梁|晋|楚|闫|法|汝|鄢|涂|钦|段干|百里|东郭|南门|呼延|归海|羊舌|微生|岳|帅|缑|亢|况|郈|有琴|梁丘|左丘|东门|西门|商|牟|佘|佴|伯|赏|南宫|墨|哈|谯|笪|年|爱|阳|佟|第五|言|福|百|姓)")
    register_type("do_surname",r"(姓|叫|免贵姓|信)")

    rn = RuleNER()
    rn.add_rule("extract_surname","*{do_surname:do_surname}{surname:surname}*")

    for ent in rn.extract_entities("你姓董哈", as_json=False):
        # print(ent)
        print("SOURCE_TEXT:", ent.source_text)
        print("ENTITY_NAME:", ent.entity_name)
        print("ENTITY_VALUE:", ent.entity_value)
        print("ENTITY_SPAN:", ent.spans)
        print("MATCH_RULES_DICT:", ent.rule_dict_hit)
        print("=" * 50)

    # def rule_parse_ner(query):
    #     res = []
    #     for ent in rn.extract_entities(query, as_json=False):
    #         if ent.entity_name == "surname":
    #             res.append({"entity":ent.entity_value,"spans":ent.spans})
    #     return res
    # print(rule_parse_ner("你姓董哈"))

    # register_type("unwanted_positive_rule21",r"(没说|没有|没)")
    # register_type("unwanted_positive_rule22",r"(申请|需要|要贷款|办理|需求)")
    # register_type("unwanted_negative_rule2",r"(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间|没通过|什么没有|啥子没有|啥玩意)")
    # rc_unwanted1 = SingleRuleCreator("keyword_zhongan_unwanted1") \
    #     .add_positive_rule("*{unwanted_positive_rule21:unwanted_positive_rule21}*?{unwanted_positive_rule22:unwanted_positive_rule22}*") \
    #     .add_negative_rule("*{unwanted_negative_rule2:unwanted_negative_rule2}*") \
    #     .add_rule_weight(10)

    # register_type("unwanted_positive_rule31",r"(申请|需要|要贷款|办理)")
    # register_type("unwanted_positive_rule32",r"(干嘛|干什么)")
    # register_type("unwanted_negative_rule3",r"(额度|成功|下来|批|出|征信|信用|没懂|没问题|怎么申请|没有时间|没问题|没时间)")
    # rc_unwanted2 = SingleRuleCreator("keyword_zhongan_unwanted2") \
    #     .add_positive_rule("*{unwanted_positive_rule31:unwanted_positive_rule31}*?{unwanted_positive_rule32:unwanted_positive_rule32}*") \
    #     .add_negative_rule("*{unwanted_negative_rule3:unwanted_negative_rule3}*") \
    #     .add_rule_weight(10)
    
    # ip = SingleIntentParse()
    # ip.add_intent("不需要",rc_unwanted1)
    # ip.add_intent("不需要",rc_unwanted2)
    # print("positive_value:{}".format(ip.calculate_intent("我真的不需要")))

    # rn = RuleNER()
    # rn.add_rule("rule1","*{positive:positive_rule}*")
    # #rn.add_rule("rule2","*{negative:negative_rule}*")
    # for ent in rn.extract_entities("你们是怎么知道的我的电话", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)

    # rn = RuleNER()
    # rn.add_rule("person", "我是{person}")
    # rn.add_rule("person", "我叫{person}")
    # rn.add_rule("person", "我的名字叫{person}")
    # rn.add_rule("info", "我的名字[叫|是]{person},我今年{age}岁,我住在{area}")
    # rn.add_rule("age", "我今年{age}岁")
    # # for ent in rn.extract_entities("我的名字是朱成军,我今年18岁,我住在合肥", as_json=True):
    # #     print(ent)
    # for ent in rn.extract_entities("我的名字是朱成军,我今年18岁,我住在合肥", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)

    # rn = RuleNER()
    # rn.add_rule("query","我要听{singer}的{song}")
    # for ent in rn.extract_entities("我要听刘德华的忘情水", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)

    # rn = RuleNER()
    # rn.add_rule("email","我的邮箱是{email:email}")
    # for ent in rn.extract_entities("我的邮箱是cjzhu3@iflytek.com", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)

    # register_type("name",r"(朱成军|张三|李四|王二麻)")
    # register_type("pron",r"(你|我|他|她|它|)")
    # register_type("refuse",r"(不要|不用|不想要|不考虑|不需要|用不着|用不上|不办贷款|不申请)")

    # register_type("disclosure_verb0",r"(怎么知道|名单|买卖信息|信息泄露|个人信息|你知道我是谁|知道我谁|我信息|我的信息|我手机|我的手机|我电话|我的电话)")
    # register_type("disclosure_verb",r"(谁给|谁告诉|哪里|得到|拿到|搞到|哪来|哪搞的|哪里来|哪里拿|为什么你们知道|为什么你知道|为什么你有|为什么你们有|为什么知道|怎么知道|怎么会有|怎么有|怎么得|怎么来|怎么拿到)")
    # register_type("disclosure_noun",r"(电话|手机|号码|信息|联系方式)")
    
    # rn = RuleNER()
    # rn.add_rule("info","*叫{person:name},我的邮箱是{email:email}")
    # rn.add_rule("rule_zhongan_material","*{disclosure_verb:disclosure_verb}*{disclosure_noun:disclosure_noun}*")
    # # rn.add_rule("info","我的名字叫{person:name},我的邮箱是{email:email}")
    # # rn.add_rule("name","我的名字[叫|是]{person}")
    # # rn.add_rule("email","我的邮箱是{email:email}")
    # # rn.add_rule("reject","{who:pron}?.*{reject:refuse}.*")

    # for ent in rn.extract_entities("我的名字叫朱成军,我的邮箱是cjzhu3@iflytek.com", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)
    #     print("=" * 50)

    # for ent in rn.extract_entities("我的名字是朱成军", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)
    #     print("=" * 50)

    # for ent in rn.extract_entities("我的名字是朱成军", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)
    #     print("=" * 50)

    # for ent in rn.extract_entities("嗯不用了我不需要用不上", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)
    #     print("=" * 50)
    
    # ####################################################################################
    # dn = DictionaryNER()
    # dn.add_dictionary("handle",["办理","调整","办一下"])
    # # for ent in dn.extract_entities("我 想 办理 五G 套餐！", as_json=True):
    # #     print(ent)
    # for ent in dn.extract_entities("我 想 办理 五G 套餐！", as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)
    ####################################################################################
    # dn2 = DictionaryNER()
    # dn2.add_dictionary("where",["谁给","谁告诉","哪里","得到","拿到","搞到","哪来","哪搞的","哪里来","哪里拿","为什么你们知道","为什么你知道","为什么你有","为什么你们有","为什么知道","怎么知道","怎么会有","怎么有","怎么得","怎么来","怎么拿到"])
    # dn2.add_dictionary("connect",["电话","手机","号码","手机号","信息","联系方式"])
    # # for ent in dn.extract_entities("我 想 办理 五G 套餐！", as_json=True):
    # #     print(ent)
    # from jieba import posseg
    
    # text = " ".join([key.word for key in posseg.cut('我的联系方式你们咋知道')])
    # print(text)
    # for ent in dn2.extract_entities(text, as_json=False):
    #     print("SOURCE_TEXT:", ent.source_text)
    #     print("ENTITY_NAME:", ent.entity_name)
    #     print("ENTITY_VALUE:", ent.entity_value)
    #     print("MATCH_RULES_DICT:", ent.rule_dict_hit)