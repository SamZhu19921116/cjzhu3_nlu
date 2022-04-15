import os
import pickle
import logging

class Slot_Engine():
    def __init__(self, model_dir, rule_dir):
        self.model_slot_module_dict = {}
        self.rule_slot_module_dict = {}
        # 若模型已存在则加载图模型否则重新构建新图模型
        if os.path.isdir(model_dir) and len(os.listdir(model_dir)):
            file_name_list = os.listdir(model_dir)
            for file_name in file_name_list:
                if file_name.endswith(".pkl"):
                    self.model_slot_module_dict[file_name.strip(".pkl")] = self.json_deserialization(os.path.join(model_dir,file_name))

        # 若模型已存在则加载图模型否则重新构建新图模型
        if os.path.isdir(rule_dir) and len(os.listdir(rule_dir)):
            file_name_list = os.listdir(rule_dir)
            for file_name in file_name_list:
                if file_name.endswith(".pkl"):
                    self.rule_slot_module_dict[file_name.strip(".pkl")] = self.json_deserialization(os.path.join(rule_dir,file_name))

    def model_slot_update(self, from_json, to_dir, to_file):
        if os.path.exists(os.path.join(to_dir, to_file)):
            os.remove(os.path.join(to_dir,to_file))
        self.model_slot_module_dict[to_file] = from_json
        self.json_serialize(os.path.join(to_dir,to_file+".pkl"), from_json)

    def rule_slot_update(self, from_json, to_dir, to_file):
        if os.path.exists(os.path.join(to_dir, to_file)):
            os.remove(os.path.join(to_dir,to_file))
        self.rule_slot_module_dict[to_file] = from_json
        self.json_serialize(os.path.join(to_dir,to_file+".pkl"), from_json)

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
