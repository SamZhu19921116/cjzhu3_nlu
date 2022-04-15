import logging
from flask import Flask, request, jsonify
from Step4_Retrival_Sort_Credit_V1 import Auto_Sort_Keyword_Semantic_V21
from bert_ner_surname.bert import Ner
model = Ner("bert_ner_surname/out_base/")
app=Flask(__name__) #创建新的开始

@app.route('/credit/keyword/')
def credit_robot_keyword():
    query = request.args.get('query',type=str)
    node = request.args.get('node',type=str)
    ner = request.args.get('ner',type=str)
    if ner == '1':
        surname_json = model.predict(query)
    else:
        surname_json = ''
    intent_json = Auto_Sort_Keyword_Semantic_V21(node,query)
    app.logger.info("user_query:{},current_node:{},is_ner:{},intent:{},surname:{}".format(query,node,ner,intent_json,surname_json))
    return jsonify({'intent':intent_json,'surname':surname_json})

if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    handler = logging.FileHandler('flask.log', encoding='UTF-8')   # 设置日志字符集和存储路径名字
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s'))
    app.logger.addHandler(handler)
    app.run(host='0.0.0.0',port=1472,debug=True) #运行开始
