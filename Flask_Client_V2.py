import urllib.parse  
import urllib.request

#urlencode可以把key-value这样的键值对转换成我们想要的格式，返回的是a=1&b=2这样的字符串
#百度搜索的页面的请求为'http://www.baidu.com/s?wd=',wd为请求搜索的内容
#urlencode遇到中文会自动进行编码转化
#一个参数时可以采用'http://www.baidu.com/s?wd='+keywd的格式，
# 但是当keywd为中文的时候需要用urllib.request.quote(keywd)进行编码转换
data = urllib.parse.urlencode({'query': '我已经贷过款了'})
response = urllib.request.urlopen('http://172.21.191.94:1472/keyword/?%s' % data)
print(response)
