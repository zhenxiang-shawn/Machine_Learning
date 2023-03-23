import requests
import re

url = "https://www.freepik.com/free-photos-vectors/soil-background?sign-up=google"
page = requests.get(url=url).text

print(page)

res = re.compile(r'src="(http.+?.jpg)"')#运用正则表达式过滤出图片路径地址
reg = re.findall(res, page)

print(f"Found {len(reg)} images.")


num = 0
for i in reg:
    a = requests.get(i)
    f = open("./img/%s.jpg" % num, 'wb') #以二进制格式写入img文件夹中
    f.write(a.content)
    f.close()
    print("第%s张图片下载完毕" % num)
    num = num+1