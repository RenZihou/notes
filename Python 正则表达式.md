# Python 正则表达式

***

```python
import re
```

* `re.match()`：从第一个字符开始匹配直到结束

	```python
	result = re.match(pattern, string, flags=0)
	result_group = result.group()  # 匹配到的组，即表达式中小括号匹配的内容，group()中传入的参数为匹配目标的序数，不传参数返回全部匹配内容
	result_span = result.span()  # 匹配到的首尾下标，以元组形式返回
	```

	贪婪匹配：`.*`；非贪婪匹配：`.*?`

	一般情况下，`.*`能匹配换行符`\n`以外的所有字符，若想要匹配全部字符，在`re.match()`中传入参数`re.S`

* `re.search()`：扫描整个字符串，返回第一个成功的匹配

* `re.findall()`：扫描整个字符串，以列表形式返回全部成功的匹配

* `re.sub()`：匹配并替换每一个子串：

	```python
	new = re.sub(pattern, repl, string, count=0, flags=0)
	```

* `re.compile()`：将正则表达式串编译为正则表达式对象，便于复用

	```python
	pattern = re.compile(pattern_str, re.S)
	```

  

