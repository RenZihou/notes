# Python 面向对象

***

[TOC]

***

## 基础知识

* 类：具有对象特征的抽象，包括名称、属性和方法

	只有在产生对象之后，对象才具有属性和方法

	对象 - 抽象 -> 类，类 - 实例化 -> 对象

* 属性与变量的区别：

	变量根据不同位置（全局，非全局）存在不同访问权限

	属性只能通过对象访问（存在宿主）且仍存在访问权限

* Python：万物皆对象

***

## 定义类

```python
class NewClass(object):
    """类注释"""
    pass
```

* 类名满足首字母大写要求

* 父类名（通过`类名.__bases__`查询）写在括号中，无继承的情况下为`object`

* 元类：由于类本身是一个对象，故类应该是被一个类实例化出的，即元类

	在类体中`__metaclass__ = 元类名`定义或继承父类的元类，可用`类名.__class__`查询。无定义时元类为`type`

* 查询类注释：`help(类名)`

	或者在命令提示行（代码目录）中通过以下方式生成网页

	```
	python -m pydoc -b
	```

* 实例化对象

	```python
	new_object = NewClass()
	```

***

## 对象属性操作

* 给对象添加新属性（或修改属性值）：`对象名.属性名 = 属性值`（采用赋值语句）

* 访问对象属性（实例属性）：`print(对象名.属性名)` 。当对象属性未被定义时会报错`AttributeError`

	访问对象全部属性：`print(对象名.__dict__)`

* 删除对象属性：`del 对象名.属性名`

* 共有属性（开头无下划线`x`）：在类内部、子类内部、模块内（类/子类访问&类实例/子类实例访问）、跨模块（`import 模块名`或`from 模块名 import *`）均可访问

* 受保护的属性（开头一个下划线`_x`）：在类内部、子类内部可以正常访问，模块内可以访问但会受到警告，模块间可以通过`import 模块名`访问但不能通过`from 模块名 import *`访问

* 私有属性（开头两个下划线`__x`）：只能在类内部定义。在类内部可以访问，子类、模块内不可访问，模块间与受保护的属性同理

* 名字重整机制：双下划线的伪私有属性`__x`会自动被重整为`_类名__x`，可以通过此名称从外部访问

* 命名规范：`x_`用于与系统内置名区分；`__x__`表示系统内置，应避免

* 只读属性（一般指实例属性）：内部自动根据环境修改，外界只能读取

	方法：先私有化（前置双下划线）再公开读（实例方法）

	优化：装饰器`@property`使得可以以属性的方法调用方法

* 内置属性：
	* 类属性：
		`__dict__`：类属性，`dict`
		`__bases__`：父类，`tuple`
		`__doc__`：注释，`str`
		`__name__`：类名
		`__module__`：定义模块
	* 实例属性：
		`__dict__`：实例属性，`dict`
		`__class__`：对应类

***

## 类属性操作

* 给类增加/修改属性：`类名.类属性 = 属性值` 或直接在类内部定义

* 访问类属性：直接访问（同对象）或通过对象访问（类中`对象名.属性名`）

	机制：通过对象下的`__class__`查找（默认指向），故若之前定义 `对象名.__class__ = 其他类名`，则无法再通过此对象查找

* 删除类属性：`del 类名.属性名`

* 定义类时 `__slots__ = (允许的属性名)` 可以强制声明属性，达到节约内存、排错的目的

***

## 类方法

```python
class MyClass(object):
	def method_a(self):  # 实例方法
		pass

	@classmethod
	def method_b():  # 类方法
		pass

	@staticmethod
	def method_c():  # 静态方法
		pass
```

* 实例方法默认第一个参数接收到一个实例（一般称为`self`）

   类方法第一个参数接收到一个类（一般称为`cls`）

   静态方法第一个参数无默认接收

* 调用方法：保证每一个方法接收到的第一个参数类型都是其需要的类型
	* 实例方法：
	
		标准调用：`实例名.方法名(其他形参)`，此处不必再传递第一个参数，解释器自动传递实例
	
		其他调用：`类名.方法名(全部参数)`（本质上将其当作一个方法处理）
	
	* 类方法：
	
		标准调用：`类名.方法名(其他形参)`
	
		通过实例调用：`实例名.方法名(其他形参)`（此时实例被忽略，实例所属的类被传递）
	
		通过子类调用：语法同标准调用，此时将子类传递
	
	* 静态方法：可以通过类或实例调用，因为它没有默认参数，故不会有自动传递
	
* 通过实例方法可以访问类属性和实例属性

   类方法只能访问类属性而无法访问实例属性

   静态方法二者均可访问，但因为是静态方法，一般不定义和类或对象相关的内容

* 方法的注释：
	```python
	"""
	方法注释
	:param 形参名: 形参类型，作用，默认值
	:return: 返回值类型，作用
	"""
	```
	
* 链式编程：将方法的返回值设为`self`，则之后的操作可以写作`MyObject.method1().method2().method3()`

* （内置）初始化方法`__init__(self, *args, **kwargs)`，会在实例化时自动调用

* （内置）属性赋值方法`__setattr__(self, key, value)`，注意由于`对象.属性 = 值`就是在调用这一方法，故其中的操作应该是直接修改`__dict__`：`self.__dict__[key] = value`

* 私有化方法：在`def __func()`，其名字重整机制与私有属性相同。

* （内置）信息格式化方法：

   `__str__(self)`，在`print(类的实例名)`时调用，用于输出对用户友好的信息

   `__repr__(self)`，在`print(类的实例名)`且未定义`__str(self)__`时被调用，亦可通过`print(repr(类的实例名))`直接获取实例的本质信息（面向开发人员）

* （内置）调用方法：`__call__(self, *args, **kwargs)`，在对象被调用时运行

* （内置）索引方法：
	* 增添/修改：`__setitem__(self, key, value)`，此后可以以字典的形式给对象的属性赋值
	
	* 查找：`__getitem__(self, item)`
	
* 删除：`__delitem__(self, key)`，此后可以通过`del 对象名[属性名]`进行删除
	
	```python
	class MyClass(object):
		def __init__(self, *args, **kwargs):
			self.cache = {}
	
		def __setitem__(self, key, value):
			print('Set')
			self.cache[key] = value
	
		def __getitem__(self, item):
			return self.cache[item]
	
		def __delitem__(self, key):
			print('del')
			del self.cache[key]
	```
	
* （内置）比较操作：参考以下代码
	
	注意：实际进行比较时，会先查找对应方法，其次会查找共轭方法（大于小于之间可以通过调换参数的方式调用，等于/不等于之间可以替换）；不支持叠加操作
	
	```python
	class MyClass(object):
		def __eq__(self, other):  # ==
			return self.x == other.x

		def __ne__(self, other):  # !=
			return self.x != other.x

		def __gt__(self, other):  # >
			return self.x > other.x

		def __ge__(self, other):  # >=
			return self.x >= other.x

		def __lt__(self, other):  # <
			return self.x < other.x
	
		def __le__(self, other):  # <=
			return self.x <= other.x
	```
	
* （内置）上下文布尔值：`__bool__(self)`，返回值被转换为布尔值

* （内置）遍历操作：

	方式一：

	```python
	class MyClass(Object):
		def __init__(self, *args, **kwargs):
			self.count = 0

		def __getitem__(self, item):
			self.count += 1
			if self.count >= maxn:  # 判定循环上限
				raise StopIteration()  # 抛出异常终止遍历
				return self.count
	
	MyObject = MyClass()
	
	for i in MyObject:
		print(i)
	```

	方式二：迭代器（优先级更高）

	```python
	class MyClass(Object):
		def __init__(self, *args, **kwargs):
			self.count = 0
	
		def __iter__(self):  # 先被调用，获取一个迭代器对象
			self.count = 0  # 重置以实现复用，没有此行则一次性
			return self
	
		def __next__(self):  # 针对迭代器获取下一个值
			self.count += 1
			if self.count >= maxn:  # 判定循环上限
				raise StopIteration  # 抛出异常终止遍历
			return self.count
	
	MyObject = MyClass()
	
	for i in MyObject:
		print(i)
	```

***

## 描述器

* 描述器是一个对象，用来操作对象的属性（增删改查）。使用实例进行操作

	作用：在修改属性的时候进行数据的过滤等

* 下面各代码将`a`定义为描述器，此后对于属性`__a`的操作直接由`print(MyObject.a)`，`MyObject.a = value`，`del MyObject.a`完成

* 法一

	```python
	class MyClass(Object):
		def __init__(self):
			self.__a = 1
	
		def get_a(self):
			return self.__a
	
		def set_a(self, value):
			self.__a = value
	
		def del_a(self):
			del self.__a
	
		a = property(get_a, set_a, del_a)
	```

* 法二

  利用`@property`，该装饰器将被修饰的函数定义成实例的一个属性，属性的值即为函数返回值

  ```python
  class MyClass(Object):
  	def __init__(self):
  		self.__a = 1
  
  	@property
  	def a(self):  # 查询时调用
  		return self.__a
  
  	@a.setter  # 属性名.setter
  	def a(self, value):  # 赋值时调用
  		self.__a = value
  
  	@a.deleter  # 属性名.deleter
  	def a(self):  # 删除时调用
  		del self.__a
  ```

* 法三

  使用内置的`__get__`，`__set__`，`__delete__`函数

  ```python
  class A(Object):
  	def __get__(self, instance, owner):  # owner是所有者的类，instance是访问该描述器的实例（通过类访问则为None）
  		return instance.__a  # `self`为共用的类，`instance`才是不同实例
  
  	def __set__(self, instance, value):
  		instance.__a = value
  
  	def __delete__(self, instance):
  		del instance.__a
  
  class MyClass(Object):
  	a = A()
  ```

* 一个实例属性的正常访问顺序（以`get`为例）：

	1. 实例对象自身`__dict__`字典
	2. 所属类对象`__dict__`字典
	3. 上层父类的`__dict__`字典
	4. 被定义的`__getattr__`方法。

	描述其能生效是因为系统自动实现`__getattribute__`方法从而改变上述查找顺序，如果自己定义了这个方法，描述器不会生效

* 调用优先级：资料描述器（实现了`get`，`set`） > 实例属性 > 非资料描述器（仅实现`get`）

***

## 装饰器的类实现

* 装饰器要求是一个可调用（`Callable`）的对象，故只需要实现`__call__`（实现装饰逻辑）即可，此外，`__init__`用于接收被装饰的函数

	```python
	class decorator(func):  # 定义装饰器
	    def __init__(self, func):
	        self.func = func  # 保存传递过来的函数

	    def __call__(self, *args, **kwargs):
	        # 此处为装饰体
	        return self.func(*args, **kwargs)

	@decorator
	def myfunc():  # 定义需要包装的方法
	    # 此处为原函数体
	    pass
	```
	
* 如果要在类装饰器上传入参数，那么`__init__`就不再接收被装饰函数，而是接受传入的参数，而`__call__`接收被装饰函数同时实现装饰逻辑

	```python
	class decorator(object):
	    def __init__(self, key=None):
	        self.key = key

	    def __call__(self, func):
	        def wrapper(*args, **kwargs):
	            # 装饰体
	            func(*args, **kwargs)
	        return wrapper

	@decorator(key=key)
	def myfunc():  # 定义需要包装的方法
	    # 此处为原函数体
	    pass
	```

***

## 内存管理机制

### 存储方面

* 万物皆对象，不存在基本数据类型
* 所有对象，都会在内存中开辟出一块空间进行存储，返回地址给外界接受（引用），可以用`id()`查看
* 对于整数和短字符，`Python`会进行缓存（即不会创建多个对象）
* 容器对象（列表，字典，元组，对象属性...）内存储的其他对象，仅仅是其他对象的引用（地址）而非其他对象本身（值）

### 垃圾回收方面

* 引用计数器：记录一个地址被引用的次数，当变为0时会被回收处理

	可通过`sys.getrefcount()`查看括号内对象所连接的地址被引用次数，注意返回的值包含这一次

* 引用次数+1：

	对象被创建

	对象被引用

	对象被作为参数传入函数（实际上由于函数内部会有两个属性持有该对象（`func_globals`，`__globals__`），故会+2）

	对象作为元素被存储在容器中

* 引用次数-1：

	对象别名被显式销毁`del 对象名`

	对象别名被赋予新的对象（地址不再被该对象引用）

	对象离开作用域（函数执行完毕，局部变量）

	对象所在容器被销毁

* 循环引用（两个容器对象互相引用）：引用计数器机制失效，只能靠效率较低的垃圾回收机制：

	1. 针对每一个容器对象记录其引用次数（`gc_refs`）
	2. 针对每一个容器对象，将其所引用的容器对象的引用次数-1
	3. 一轮循环后引用次数为0的即为循环引用的产物，可以清除

* 分代回收：为了提高垃圾回收机制效率，将容器对象分为0代、1代、2代，被检查一次且不被清除的会被划分到更高一代，代数越高，检测频率越低

	可用`gc`模块中`gc.get_threshold()`进行查看，所返回的元组中三个参数分别为触发检测机制的`新增对象值-消亡对象值`的阈值、0代1代频率之比、1代2代频率之比

	可用`gc.set_threshold()`设置

* 垃圾回收机制的触发：自动触发：条件：`gc.isenabled()`为`True`（可用`gc.enable()`以及`gc.disable()`设置），且`新增对象值-消亡对象值`达到阈值。手动触发：在`del`后，使用`gc.collect()`

***

## 封装

* 将一些属性和方法封装在一个对象中，而隐藏内部具体细节。
* 优点：可以设置属性为私有以保证安全，也可以对数据进行拦截校验，便于维护
* 实现：`from ... import ...`

***

## 继承

### 基础知识

* 一个类拥有（获得使用权）另一个类的资源（非私有的属性和方法）的方式之一

* 目的：将重复的内容整合

* 实现：

	```python
	class BaseClass1(object):
	    pass
	
	class BaseClass2(object):
	    pass
	
	class SubClass(BaseClass1, BaseClass2):  # 有顺序
	    pass
	```

* 查询：`类名.__base__`

* 父类与元类：`type`实例化出`object`，同时`type`继承自`object`

### 历史发展

* 可以被继承的属性：类的公有属性和受保护的属性

	只有使用权，故使用同一块内存空间

	对子类的属性进行赋值时，会在子类中创建一个新的属性

* 三种继承形态：

	1. 单继承：一级级继承，访问资源的时候会一级级往上找

	2. 无重叠的多继承：继承自两个无重叠的单继承链（`A`继承自`B`、`C`，而`B`、`C`分别继承自`D`、`E`）

		访问资源遵循_单调原则_：按照继承顺序，找完一条链再找下一条

	3. 有重叠的多继承：`A`继承自`B`、`C`，而`B`、`C`均继承自`D`

		访问资源遵循_重写可用原则_：先找`B`、`C`，再找`D`

* `Python 2.1`

	只有经典类，采用`MRO`算法，遵循_深度优先_：

	1. 从根节点开始压栈（栈：先入后出）
	2. 出栈（同时将其所对的上一级压栈，重复则跳过）
	3. 循环

	问题：无法正确处理 有重叠的多继承

* `Python 2.2`

	出现新式类，在新式类的继承中优化：

	如果压栈时遇到重复，则跳过之前的（即所谓的入列（列：先进先出）、出列，在 无重叠的单继承链 上不同）

	问题：处理类似于“`A`继承自`B`、`C`，`C`又继承自`B`”（有问题的继承）时，会先查找`C`再查找`B`，违反_局部优先原则_

* `Python 2.3 - 2.7`

	新式类采用`C3`算法。两个公式：

	1. `L(object) = [object]`
	2. `L(子类(父类1, 父类2)) = [子类] + merge(L(父类1), L(父类2), [父类1, 父类2])`

* `merge`算法：

	1. 判断第一个列表的第一个元素是后续列表的第一个元素或不在后续列表中出现

		* 为真则将其加入到最终解析列表中并从当前操作的全部列表中删除

		* 为假则跳过此元素，查找下一个列表的第一个元素

	2. 重复判断

	3. 若最终无法把全部元素归并到解析列表中，则报错

* `C3`算法与拓扑算法（找入度为0的节点，输出并删除该节点及其出边，重复），在判断错误继承的时候不同

* `Python 3.x`：不再存在经典类，统一使用`C3`算法

* 查看继承优先级：`类名.mro()`

### 资源的覆盖

* 资源的覆盖：优先级较高的覆盖优先级低的
* 继承方法时，`cls`和`self`的变化：谁调用的这一方法，就代表谁

### 资源的累加

* 资源的累加：在父类的基础之上，多了一些自己特有的资源

* 资源的累加场景1：初始化方法

	当子类有初始化方法的时候，子类产生对象时不会调用父类的初始化方法（子类无初始化方法时会调用），从而不会有父类初始化方法中的属性

* 资源的累加场景2：方法的累加

	1. 通过类调用父类的方法：`A`继承自`B`，可在`A`的方法中加一行：

	* `B.实例方法(self)`（此时`self`代表了A所创建出的实例）
	* `B.类方法(cls)`（此时`cls`代表了`A`）
	* `B.静态方法()`

	存在的问题：多级继承时会导致重复调用

	2. 沿着`MRO`链条找到下一级节点，去调用相应的方法

		语法原理：

		```python
		def super(cls, inst):
			mro = inst.__class__.mro()
			return mro[mro.index(cls) + 1]
		
		# Python 2.x
		super(type, obj).method()  # method()为要调用的实例方法，type在此时为所处类名或其子类类名（这样才能在mro链条中找到该类），obj为创建的实例对象，即self
		super(type, obj).class_method()  # class_method()为要调用的类方法，type在此时为所处类名或其子类类名（这样才能在mro链条中找到该类），obj为类本身，即cls
		
		# Python 3.x
		super().method()  # method()为要调用的方法，Python 3 会根据所处环境自动添加参数
		```

***

## 多态

* 一个类所具有的多种形态，或调用时的多种形态

* 实现：在子类中进行重写

	```python
	class BaseClass(object):
		def func(self):
			pass
	
	class SubClass1(BaseClass):
		def func(self):
			# code1
			pass
	
	class SubClass2(BaseClass):
		def func(self):
			# code2
			pass
	
	def test(obj):
		obj.func()
	
	a = SubClass1()
	b = SubClass2()
	test(a)
	test(b)
	```

* 由于`Python`是动态语言，只要传入`test`的对象具有方法`func()`，代码就可以正常运行，故`Python`中不存在真实意义上的多态（即不需要像静态语言一样传入类型参数），也不需要实现多态

***

## 抽象类与抽象方法

* 抽象类：一个抽象出来的类，不是具化的，也不可以直接创建实例

	抽象方法：一个抽象出来的方法，不具备具体实现，也不可以直接调用

* 意义：可以作为父类存储子类所共有的资源，但自身不能产生实例

* 实现：

	```python
	import abc
	
	class NewClass(object, metaclass=abc.ABCMeta):
	    @abc.abstractmethod
	    def abstractmethod():  # 抽象方法
	        pass
	```

* 想让抽象类的子类能够使用，必须实现抽象类中所有抽象方法

***

## 类的设计原则——SOLID

* S - _Single Responsibility Principle 单一职责原则_

	一个类只负责一项职责，便于代码维护

* O - _Open Closed Principle 开放封闭原则_

	对扩展开放，对修改关闭

* L - _Liskov Substitution Principle 里氏替换原则_

	使用基类引用的地方必须能使用继承类的对象，即继承类必须完全继承基类的资源

* I - _Interface Segregation Principle 接口分离原则_

	当一个类包含太多的接口（抽象方法），并且这些方法在使用过程中并非“不可分割”，就应适当将其分离

* D - _Dependency Inversion Principle 依赖倒置原则_

	高层模块不应直接依赖低层模块，而应依赖抽象类或接口