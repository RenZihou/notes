# Python 爬虫

***

* [测试网站](https://httpbin.org)

## 原理

* 浏览器向服务器发送`HTTP`请求（`Request`），请求包含：
	1. 请求方式（`Get`（`url`中的`parameter string`，如：查询`search`），`Post`（需要构造表单，然后提交，请求参数不包含在`url`中，如：登陆））
	2. 统一资源定位器`url`
	3. 请求头`headers`（包含浏览器信息，帮助服务器判断请求的合法性）
	4. 请求体（额外携带的数据，如表单提交时的表单数据）

* 服务器的响应（`Response`）包含：
	1. 状态码`status_code`（`200`表示正常，`300`以上表示跳转，`400`以上表示用户端错误（如`404 Not Found`），`500`以上表示服务器错误）
	2. 响应头（内容类型，设置`Cookie`等）
	3. 响应体（包含请求资源的内容）

* 检查元素中看到的源代码时经过`js`渲染的，不同于网页源代码

## 第三方模块

```python
import requests  # 请求
from lxml import etree  # 解析
```

## Get 请求

* 基本用法：

	```python
	response = requests.get(url)
	```

	`response`的属性：`status_code`，`headers`，`cookies`，`url`，`history`

* 添加请求头：
	
	```python
	headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}  # 请求头
	response = requests.get(url, headers=headers)
	```
	
* 获取二进制码：

	```python
	requests.get(url, headers=headers).content
	```

	可直接写入文件（`wb`模式）

* 使用`cookies`进行会话维持（否则每一次请求都是独立的，`cookies`不会保存）：

	```python
	s = requests.Session()
	```

* 证书验证：默认开启，验证不安全时会报错。关闭：

	```python
	response = requests.get(url, verify=False)
	```

* 代理设置：
	```python
	proxies= {'http': 'http://127.0.0.1:9743', 'https': 'https://127.0.0.1:9743'}  # 如果有密码，使用http://user:password@127.0.0.1:9743/格式
	response = requests.get(url, proxies=proxies)
	```
	
* 超时设置：

	```python
	requests.get(url, timeout=1)  # 以秒为单位
	```

* 认证设置：
	```python
	from requests.auth import HTTPBasicAuth
	response = requests.get(url, auth=HTTPBasicAuth(user, password))
	```

***

## 解析JSON

* 法一：

	```python
	requests.get(url, headers=headers).json()
	```

* 法二：

	```python
	import json
	json.loads(requests.get(url, headers=headers).text)
	```

***

## Post请求

* 提交表单：

	```python
	data = {'username': 'user1', 'password': 'pwd123'}
	response = requests.post(url, data=data, headers=headers)
	```

* 上传文件：

	```python
	files = {'file': open(file_name, 'rb')}
	response = requests.post(url, files=files)
	```

***

## etree解析

```python
data = requests.get(url, headers=headers).text # 下载网页信息
s = etree.HTML(data) # 解析网页
element = s.xpath('元素的xpath信息/text()') # 在Chorme中在元素处 右键-检查-Copy-Copy XPath，当出现tbody（表示表格本体）时手动删除
```

***

## 细节

* 防反爬基础操作：延迟请求

	```python
	from time import sleep
	sleep(2)  # 以秒为单位
	```

	