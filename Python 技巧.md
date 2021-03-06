# Python 技巧

***

[TOC]

***

## PyCharm

* 代码模板

	`File` - `Settings` - `Editor` - `File and Code Templates` - `Python Script`

* 预设代码块

	`Ctrl + J`，可以调用各种常见代码块

***

## Pythonic

* 访问字典元素

	```python
	dic = {'a': 1, 'b': 2}
	print(dic.get('a', 'Not Found'))  # 1
	print(dic.get('c', 'Not Found'))  # Not Found
	```

* 操作列表

	筛选符合某些条件的值：

	```python
	a = [3, 4, 5]
	b = [i for i in a if i >= 4]  # b = [4, 5]
	c = list(filter(lambda x: x % 4 == 0, a))  # c = [4]
	```

	对每个元素进行运算：

	```python
	a = [3, 4, 5]
	b = [i + 3 for i in a]  # b = [6, 7, 8]
	c = list(map(lambda i: i + 3, a))  # c = [6, 7, 8]
	```

* 链式比较

	```python
	a = 2
	print(1 < a < 3)  # True
	```

* 一次实现多个赋值

	```python
	a, b = b, a
	```

* 占位符

	```python
	filename = 'file.txt'
	name, _, ext = filename.rpartition('.')  # name = 'file', ext = 'txt'
	```

	不会产生名为`_`（接收了`rpartition()`返回的第二个值，即`.`）的变量

* 序列解包

	使用`*`来获取多个元素

	```python
	li = [1, 2, 3, 4]
	a, *b = li  # a = 1, b = [2, 3, 4]
	c, *d, e = li  # c = 1, d = [2, 3], e = 4
	li2 = [1, 2]
	*f, g, h = li2  # f = [], g = 1, h = 2
	```

* 字符串/列表反序

	```python
	s = 'abcd'
	li = [1, 2, 3]
	print(s[::-1])  # 'dcba'
	print(li[::-1])  # [3, 2, 1]
	```

* 用`or`实现多重选择：

	`or`的原理：从第一个开始判断，出现的第一个`True`值会被返回（后面的被短路）

	故：当我们要依次调用`func1()`、`func2()`、`func3()`，要求是只有当前一个返回`None`时才调用下一个，则可以如下写：

	```python
	result = func1() or func2() or func3()
	```

***

## 提升效率

* 测试代码时间

	```python
	from timeit import timeit
	
	def fibonacci(n: int) -> int:
	    if n in (1, 2):
	        return 1
	    else:
	        return fibonacci(n - 1) + fibonacci(n - 2)
	
	print(timeit('fibonacci(40)', globals={'fibonacci': fibonacci}, number=10))
	```

	运行第一个参数里的语句`number`次，计算时间（返回总时间）。注意需要指定变量的作用域（如果上面没有`globals`参数就会报错）

* 按照调用分析运行时间

	```python
	from profile import run
	
	def fibonacci(n: int) -> int:
	    if n in (1, 2):
	        return 1
	    else:
	        return fibonacci(n - 1) + fibonacci(n - 2)
	
	run('fibonacci(40)', globals={'fibonacci': fibonacci}, number=10)
	```

* 保存函数运行结果，避免重复调用时耗费大量时间（在递归中尤其有用）

	```python
	from functools import lru_cache
	
	@lru_cache()
	def fibonacci(n: int) -> int:
	    if n in (1, 2):
	        return 1
	    else:
	        return fibonacci(n - 1) + fibonacci(n - 2)
	```

* 使用`set`而非`list`进行查找

	`set`是一个无序的无重复的集合，可以自动将列表中的重复元素去除。若要排序，如下：

	```python
	li = [1, 2, 2, 8, 5, 8]
	li2 = list(sorted(set(li)))  # li2 = [1, 2, 5, 8]
	```

* 使用`NumPy`中的函数代替`math`中的函数

	* 使用`np.array`代替`list`（向量化）

	* 使用`np.where`代替`if`

		```python
		import numpy as np
		
		array = np.arange(-10e5, 10e5)
		func_slow = np.vectorize(lambda x: x if x > 0 else 0)  # np.wectorize()用于将普通函数转化为支持向量化的函数
		func_fast = lambda x: np.where(x > 0, x, 0)
		```

		`np.where()`的第一个参数为条件，满足条件则输出第二个参数，否则输出第三个参数

* 使用`filter()`、`map()`代替推导式

	```python
	def func_slow():
	    return [x ** 2 for x in range(1, 10000)]
	
	def func_fast():
	    return map(lambda x: x ** 2, range(1, 10000))
	```

	```python
	def func_slow():
	    return [x for x in range(1, 10000) if x % 3 == 0]
	
	def func_fast():
	    return filter(lambda x: x % 3 == 0, range(1, 10000))
	```

	`map()`将第二个参数（序列）中的每一项代入第一个参数（函数，即一个映射），所有返回值构成新的结果

	`filter()`将第二个参数（序列）中的每一项代入第一个参数（函数）中，返回值为`True`则将其添加到结果中

* 计数：`collections.Counter`

	```python
	from collections import Counter
	
	li = [1, 2, 2, 2.0, 3, 4, 4]
	count = Counter(li)  # Counter({2: 3, 4: 2, 1: 1, 3: 1})
	```

	`Counter`不区分数学上相等的整型和浮点型。

	常用的方法：

	```python
	print(count.get(2))  # 获取2出现的次数，输出：3
	print(counter.most_common(2))  # 最常见的2个，输出：[(2, 3), (4, 2)]
	```

***

## 数据处理

* 数据预览

	使用`pandas-profiling`模块中的`ProfileReport()`方法，生成一份完整的数据报告，包含特征信息、数据缺陷警告、简单分析、关联、样例等

	```python
	import seaborn as sns
	import pandas_profiling
	
	titanic = sns.load_dataset('titanic')  # 使用经典的泰坦尼克获救数据集
	profile = pandas_profiling.ProfileReport(titanic)  # 分析
	profile.to_file(output_file='output_file.html')  # 保存到html文档中
	```
***

## 一些库

### Itertools

```python
import itertools
```

* 一维化列表/元组

	```python
	a = [[1, 2], [3, 4], [5, 6]]
	b = list(itertools.chain.from_iterable(a))  # b = [1, 2, 3, 4, 5, 6]
	```

* 创建笛卡尔积

	```python
	a = list(itertools.product([1, 2, 3], [4, 5]))  # a = [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
	```

* 创建排列$A_m^n$

	```python
	a = list(itertools.permutations([1, 2, 3, 4], 3))  # 对[1, 2, 3, 4]中的元素进行4A3排列
	```

* 创建组合$C_m^n$

	```python
	a = list(itertools.combinations([1, 2, 3, 4], 3))  # # 对[1, 2, 3, 4]中的元素进行4C3组合
	```

### Typing

```python
from typing import *
```

* 声明容器对象内部元素类型：

  ```python
  names: List[str] = ['Tom', 'Jack']
  classes: Tuple[int] = (2, 5)
  scores: Dict[str, int] = {'Tom': 99, 'Jack': 85}
  ```

* 推荐用法

	```python
	# Dict注释返回类型，Mapping注释参数
	def size(rect: Mapping[str, int]) -> Dict[str, int]:
	    return {'area': rect['width']*rect['height'], 'perimeter': 2*(rect['width']+rect['height'])}
	
	# Set注释返回类型，AbstractSet注释参数
	def describe(s: AbstractSet[int]) -> Set[int]:
	    return set(s)
	```

* 定义特殊含义的类型：

	```python
	Person = NewType('Person', Tuple[str, int, float])
	person = Person(('Mike', 22, 1.75))
	```

* `Union`：`Union[int, float]`表示整型和浮点型均可，常用于声明参数

* `Optional`：`Optional[str] == Union[str, None]`，常用于声明参数（传递`None`并不代表不传参数）

### Tqdm

* 在顶层`for`循环上加一个`tqmd()`函数以实现进程监控

	```python
	from time import sleep
	from tqdm import tqdm
	
	for each in tqdm(range(1000)):
	    sleep(.5)
	```
