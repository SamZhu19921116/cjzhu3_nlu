import pickle
# json字符串反序列化
def json_deserialization(from_file):
    with open(from_file, 'rb+') as f:
        data = pickle.loads(f.read())
        return data

import os
hnsw_model_dir = "/root/cjzhu3_nlu/hnsw_model"
file_name_list = os.listdir(hnsw_model_dir)
for file_name in file_name_list:
    if file_name.endswith(".pkl"):
        json_des = json_deserialization(os.path.join(hnsw_model_dir,file_name))
        print("file_name:{},id_sentence_dict_len:{}".format(file_name,len(json_des["id_sentence_dict"])))

# json_des = json_deserialization("/root/cjzhu3_nlu/rules_model/NLU_V1.0_还呗_知识库_XF04PE.pkl")
# print(json_des)
# sum = 0
# for key,value in json_des.items():
#     sum += len(value)
# print(sum)

rule_model_dir = "/root/cjzhu3_nlu/rules_model"
file_name_list = os.listdir(rule_model_dir)
for file_name in file_name_list:
    if file_name.endswith(".pkl"):
        json_des = json_deserialization(os.path.join(rule_model_dir,file_name))
        sum = 0
        for key,value in json_des.items():
            sum += len(value)
        print("file_name:{},rule_len:{}".format(file_name,sum))