from flask import Flask, request, jsonify
from Step4_Retrival_Sort_ZhongAn_V2 import Auto_Sort_Keyword_Semantic_V2
from Step4_Retrival_Sort_ZhongAn_V2 import Auto_Sort_Rule_Semantic_V1
from Step4_Retrival_Sort_ZhongAn_V2 import Auto_Sort_Rule_Semantic_V2
from Step4_Retrival_Sort_Credit_V1 import Auto_Sort_Keyword_Semantic_V21
from bert_ner_surname.bert import Ner
model = Ner("bert_ner_surname/out_base/")
app=Flask(__name__) #创建新的开始

@app.route('/') #路由设置
def index(): #如果访问了/则调用下面的局部变量
   return request.args.__str__()

@app.route('/loan/keyword/')
def robot_keywords():
   query = request.args.get('query',1,type=str)
   print("query:{}".format(query))
   parse_list = Auto_Sort_Keyword_Semantic_V2(query=query)
   finesort_intent_node = parse_list[0]
   finesort_intent_match_query = parse_list[1]
   keyword_score = parse_list[2]
   sbert_score = parse_list[3]
   w2v_score = parse_list[4]
   return jsonify({'query':query,'node':finesort_intent_node,'match':finesort_intent_match_query,'kscore':str(keyword_score),'sscore':str(sbert_score),'wscore':str(w2v_score)}), 200, {"Content-Type":"application/json"}

@app.route('/loan/rule/')
def robot_rule():
   query = request.args.get('query',1,type=str)
   print("query:{}".format(query))
   intent_json = Auto_Sort_Rule_Semantic_V2(query=query)
   surname_json=model.predict(query)
   return jsonify({'intent':intent_json,'surname':surname_json})

@app.route('/credit/keyword/')
def credit_robot_keyword():
    query = request.args.get('query',type=str)
    node = request.args.get('node',type=str)
    ner = request.args.get('ner',type=str)
    if ner == '1':
        surname_json = model,predict(query)
    else:
        surname_json = ''
    intent_json = Auto_Sort_Keyword_Semantic_V21(node,query)
    return jsonify({'intent':intent_json,'surname':surname_json})
    
if __name__ == '__main__':
   app.config['JSON_AS_ASCII'] = False
   app.run(host='0.0.0.0',port=1472,debug=True) #运行开始
