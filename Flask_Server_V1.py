from flask import Flask, request, jsonify
from Step4_Retrival_Sort_ZhongAn_V2 import Auto_Sort_Keyword_Semantic_V2
app=Flask(__name__) #创建新的开始

@app.route('/') #路由设置
def imdex(): #如果访问了/则调用下面的局部变量
   return request.args.__str__()

@app.route('/<query>')
def robot_keywords(query):
   print("query:{}".format(query))
   parse_list = Auto_Sort_Keyword_Semantic_V2(query=query)
   finesort_intent_node = parse_list[0]
   finesort_intent_match_query = parse_list[1]
   keyword_score = parse_list[2]
   sbert_score = parse_list[3]
   w2v_score = parse_list[4]
   return jsonify({'query':query,'node':finesort_intent_node,'match':finesort_intent_match_query,'kscore':str(keyword_score),'sscore':str(sbert_score),'wscore':str(w2v_score)}), 200, {"Content-Type":"application/json"}

if __name__ == '__main__':
   app.config['JSON_AS_ASCII'] = False
   app.run(host='0.0.0.0',port=1472,debug=True) #运行开始
