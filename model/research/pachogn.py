import  requests


# r = requests.get("https://zhuanlan.zhihu.com/p/660522735")

r = requests.get("http://dev.golang123.com/")
print(r.text)
