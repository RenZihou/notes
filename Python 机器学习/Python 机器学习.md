# Python 机器学习笔记

***

[TOC]

***

## 概述

* 有监督学习：
	* 分类问题——将样本归类到已知的若干个类别中
	* 回归问题——根据样本特征预测结果
* 无监督学习：
	* 聚类问题——挖掘数据的关联模式
	* 强化问题——研究如何基于环境而行动，寻求最优解策略
* 强化学习

***

## Numpy库的使用

```python
import numpy as np
```

* 读取文件：

	```python
	df = np.genfromtxt(text_file_name)
	```

	参数详见`help()`

* 创建数组：

	```python
	matrix = np.array([[1, 2, 3], [4, 5, 6]])  # 传入一个列表（可以是多维的）
	```

* 数组的属性：

	* `array.shape`，表示数组中的元素个数（多维的则表示不同维度上元素的个数）
	* `array.dtype`，表示数组中的元素类型（所有元素必须是同一类型）

* 获取数组中的元素：下标，切片；布尔类型也可以当作索引：

	```python
	new_matrix = matrix[matrix == 5]  # 会返回含有元素5的数组
	```

* 执行运算：对每一个元素进行操作

* 转换类型：`matrix.astype(float) `，传入参数类型

* 求极值：`min()`，`max()`，`matrix.argmax(axis=1)` （求每行的最大值，返回一个数组）

* 求和：`matrix.sum()`，不传参数全部求和，传入`axis=0`对列求和，传入`axis=1`对行求和

* 变换格式：

	```python
	vector.reshape(2, 3)  # 变成一个二行三列的矩阵，如果第二个参数传入-1，则自动计算
	```

* 初始化：

	```python
	matrix = np.zeros((3, 4), dtype=np.int32)  # 创建一个三行四列，值全为0（整型）的矩阵
	matrix = np.ones((3, 4))  # 创建一个三行四列，值全为1的矩阵
	```

* 按照步长创建：

	```python
	vector = np.arange(0, 12, 2)  # [0, 2, 4, 6, 8, 10]
	vector = np.linspace(0, 10, 6, dtype=int32)  # [0, 2, 4, 6, 8, 10]
	```

* 随机函数：

	```python
	matrix = np.random.random((3, 4))  # 创建一个三行四列，值在-1和1之间随机的矩阵
	```

* 矩阵乘法：`A.dot(B)`或`np.dot(A, B)`

* 矩阵拼接：

	```python
	np.hstack((A, B))  # 横向拼接
	np.vstack((A, B))  # 纵向拼接
	```

* 矩阵切分：

	```python
	np.hsplit((A, 3))  # 横向切成3份
	np.vsplit((A, 2))  # 纵向切成2份
	np.hsplit((A, (3, 4)))  # 指定刀位：在第三列和第四列后切开
	```

* 复制：`b = a`会创建同一个对象的两个链接，即`a`和`b`是一个东西；`c = a.view()`会创建元素相同的另一个对象（共用了相同的值，且一直保持这样的联系）；`d = a.copy()`相当于用`a`的值对`b`进行初始化，之后不再关联

***

## Pandas库的使用
```python
import pandas as pd
```
* 读取文件：得到一个`DataFrame`类型

	```python
	df = pd.read_csv(file_name, header=None, names=('a', 'b', 'x', 'y'))
	```

	对于第一行为列名的表格，不需要传入`header`和`names`参数，但对于第一行就是数据的表格，给`header`传入`None`，给`names`传入每列的标签，从而达到自定义列标签的目的

* `DataFrame`的属性：

	* `df.columns` ，列名，后加`.tolist()`可以返回列表
	* `df.shape` ，`DataFrame`的结构：`(样本数，标签数)`
	* `df.loc[i]`，第i行数据

* `DataFrame`的方法：

	```python
	df.head(n)  # 取前n条数据
	df.tail(n)  # 取后n条数据
	```

* 切片：`df[列名]`

* 运算：对列中的全部元素进行同一操作（与`numpy`同）

* 组合列运算：如果维度一样，在对应位置进行操作

* 求列中的最大值/最小值：`df[列名].max()`

* 求列中的均值：`df[列名].mean()`

* 排序：`df_sorted = df.sort_values(用于排序的列名)`

	重新标注行号：`df_sorted_reindexed = df.sorted.reset_index(drop=True)`，`drop`表示是否舍弃原行号（如果传入`False`，会新添一列`index`，值为原行号）

* 处理缺失值`NaN`：

	```python
	df_is_null = pd.isnull(df[target_column])  # 返回True和False组成的Frame，表示target_column（列名）这一列是否为空值
	df_null_false = df[df_is_null == False]  # 不含缺失值的Frame
	df_not_null = df.dropna(axis=0, subset=[])   # axis表示清除行还是列（0为行，1为列），subset表示要清理那些列的空值
	```

* 筛选子集：`subset = df[筛选条件]`

	例：筛选出`a`值为1的样本构成子集

	```python
	subset_a_is_1 = df[df['a'] == 1]
	```

* 透视表：

	```python
	df.pivot_table(index=None, values=None, aggfunc='mean')
	```

	`index`是用于分组的列标签，`values`是统计值，`aggfunc`是统计函数

	添加多个标签时使用列表或元组表示，常见的统计函数有：`mean`，`sum`，`max`，`min`，`count`

* 自定义函数：`df.apply(user_func)` ，通常会传入匿名函数

	例：返回每一列非空值的个数

	```python
	df.apply(lambda x: len(x[pd.isnull(x) == False]))
	```

* `DataFrame`内部结构是`Series`，而`Series`内部结构是`ndarray`

## Matplotlib库的使用
```python
import matplotlib.pyplot as plt
```
* 绘制折线图：`plt.plot(x_axis, y_axis)`，传入`x`轴和`y`轴的列表/元组

* 调整坐标标签：`plt.xticks(rotation=0)`，对于`plt.yticks()`同理

	例：

	```python
	plt.xticks(np.arange(5), ('Tom', 'Dick', 'Harry', 'Sally', 'Sue'), rotation=40)
	```

* 添加标题、坐标轴信息：

	```python
	plt.xlabel('x-axis')
	plt.ylabel('y-axis')
	plt.title('title')
	```

* 绘制多图：

	```python
	fig = plt.figure(figsize=(l, w))  # figsize表示长和宽
	new_plot = fig.add_subplot(r, c, x)  # r为矩阵行数，c为矩阵列数，x为所绘新图的位置（一行一行数，第几个）
	```

* 添加图例：`plt.legend(loc='best')`，注意在绘制每条曲线的时候明确`color`和`label`

* 绘制条形图：

	`plt.bar(x, height, width=0.3)`，前两项传入横坐标和纵坐标，列表/数组形式，`width`为柱宽度

	`plt.barh()`绘制横向的条形图，参数一样

* 绘制散点图：`plt.scatter(x, y)`

* 绘制频数分布直方图：`plt.hist(x, bins=10)`，`x`是值的集合，`bins`表示平均分为多少组

* 一般画图步骤：

	```python
	fig, ax = plt.subplots(figsize=(10, 5))  # 传入绘图尺寸大小
	ax.scatter(data1['x'], data1['y'], s=30, c='b', marker='o', label='data1')  # 用data1绘制散点图，大小30，蓝色，每个数据点用圆点标识、
	ax.scatter(data2['x'], data2['y'], s=30, c='r', marker='x', label='data2')  # 用data2绘制散点图，大小30，红色，每个数据点用叉号标识
	ax.legend()  # 图例
	ax.set_xlabel('x')  # x轴名称
	ax.set_ylabel('y')  # y轴名称
	```

***

## Seaborn库的使用
```python
import seaborn as sns
```
* 设置默认风格：`sns.set()`
	* `sns.set_style('white')`：背景白，无参考线
	* `sns.set_style('whitegrid')`：背景白，有纵坐标参考线
	* `sns.set_style('dark')`：背景浅蓝，无参考线
	* `sns.set_style('darkgrid')`：背景浅蓝，有纵坐标参考线
	* `sns.set_style('ticks')`：背景白，无参考线，坐标有短线
	
* （单变量）直方图：`sns.distplot(data)`，可选参数：`kde`拟合曲线，`hist`柱形条，`bins`分成的组数

* （二维变量）散点图：
		
	```python
	sns.scatterplot(x=x, y=y)  # 一种传参方法
	sns.scatterplot(x=x_name, y=y_name, data=pd.DataFrame(data=ori_data, columns=(x_name, y_name)))  # 另一种传参方法，其中ori_data为原始二维数据
	```
	
	或：
		
	```python
	sns.jointplot(x=x_name, y=y_name, data=pd.DataFrame(data=ori_data, columns=(x_name, y_name)))
	```
	
	可选参数：`kind`：传入`scatter`为标准散点图，`hex`为六边形热力图，其他见文档。能同时绘制散点图和直方图
	
* （二维变量）回归图：`sns.regplot()`，传入的参数同上，可选参数`x_jitter`为`x`方向随机抖动偏差

* 盒图处理离群点：`sns.boxplot()`，传入的参数同上，可选参数`hue`根据某个维度（标签）分类，`orient`为取向，图中的横线为四分位，菱形为离群点

* 小提琴图：`sns.violinplot()`，传入的参数同上，可选参数`hue`同上，`split`为是否按照分类标准将每个“小提琴”分为两部分

* 点图描述差异性变化：`sns.pointplot()`，传入的参数同上，可选`hue`

* 分类属性绘图：`sns.factorplot()`，传入参数同上，可选参数`kind`为图的种类，`col`为列的分类标签（`row`为行）

* 分类属性绘制多图：
	```python
	g = sns.FacetGrid(data, col=绘图分类标签, hue=图例分类标签)  # data应为DataFrame格式
	g.map(绘图函数, 传入绘图函数的变量)  # 传入的参数按顺序写就行，无需写在一个容器对象内
	g.add_legend()  # 显示图例
	```
	
* 热力图：
	```python
	data = df.pivot(y_label, x_label, value_label)  # 将DataFrame转换为需要的矩阵形式
	sns.heatmap(data)  # 可选参数vmin/vmax调色板的上下界，center调色板中心
	```

***

## 线性回归算法

* 目标：建立从$X$到$y$的映射，解决回归问题

* 拟合：
	$$
	h(\theta) = \sum_{i=0}^n \theta_ix_i = \theta^Tx
	$$
	其中$θ$表示参数，$θ_0$为偏置项，$x$表示特征，$x_0 = 1$，$n$为特征数，$θ^T$表示矩阵

* 误差：$ε$，对于每个样本由=有：
	$$
	y^{(i)} = \theta^Tx^{(i)}+\epsilon^{(i)}
	$$
	对于自然体系服从高斯分布
	$$
	P(y^{(i)}|x^{(i)};\theta) = \frac{\exp({-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}})}{\sqrt{2\pi}\sigma}
	$$
	此处$x^{(i)}$和$y^{(i)}$表示不同样本

* 似然函数：
	$$
	L(\theta) = \prod_{i=1}^m P(y^{(i)}|x^{(i)};\theta)
	$$
	对数似然：
	$$
	\begin{align}
	\log L(\theta) &= \sum_{i=1}^m \log P(y^{(i)}|x^{(i)};\theta) \\ 
	&= m\log \frac{1}{\sqrt{2\pi}\sigma} - \frac{\sum_{i=1}^m (y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2}
	\end{align}
	$$
	$m$为样本数

* 最小二乘法：让似然函数（对数似然）越大越好，即
	$$
	J(\theta) = \frac{1}{2} \sum_{i=1}^m (y^{(i)}-\theta^Tx^{(i)})^2 
	= \frac{1}{2}(X\theta-y)^T(X\theta-y)
	$$
	越小越好，通过偏导数得到
	$$
	\theta = (X^TX)^{-1}X^Ty
	$$

* 评估：
	$$
	R^2 = 1-\frac{残差平方项}{类似方差项} = 1 - \frac{\sum_{i=1}^m (\hat{y_i}-y_i)^2}{\sum_{i=1}^m (y_i-\bar{y})^2}
	$$

* 梯度下降原理，求$J(\theta)$最小值（逆着梯度方向，一步步迭代（步长，即学习率要小，但可以变））

	三种方法：

	* 批量（考虑全部样本，准但慢）
	* 随机（每次选一个样本（32，64，128），快但不一定准）
	* 小批量（每次选部分样本）

***

## 逻辑回归算法

* 目标：在空间中找到一个分类边界，解决二分类问题

	本质上是在线性回归的基础上加了一层非线性的`Sigmoid`映射

### 数学推导

* `Sigmoid`函数：
	$$
	g(z) = \frac{1}{1+e^{-z}}
	$$
	将输入值（实数域）映射到$(0,1)$区间上，转变为概率

* 对于二分类任务$(0, 1)$，
	$$
	\begin{align}
	P(y|x;\theta) &= (h_\theta(x))^y(1-h_\theta(x))^{1-y}\\
	h_\theta(x) &= g(\theta^Tx)
	\end{align}
	$$
	满足：
	$$
	\begin{align}
	P(y=1|x;\theta) &= h_\theta(x) \\
	P(y=0|x;\theta) &= 1-h_\theta(x)
	\end{align}
	$$
	和归一化条件

* 将概率函数代入对数似然，得到
	$$
	\log L(\theta) = \sum_{i=1}^m (y_i\log(h_\theta(x_i))+(1-y_i)\log(1-h_\theta(x_i)))
	$$
	再引入
	$$
	J(\theta) = -\frac{\log L(\theta)}{m}
	$$
	转化为梯度下降问题

* 将$J(θ)$对$θ_j$求变分，得到下降方向：
	$$
	\frac{\delta}{\delta \theta_j}J(\theta) = \frac{\sum_{i=1}^m (h_\theta(x_i)-y_i)x_i^j}{m}
	$$
	其中$i$表示样本序号，$j$表示特征序号，$m$为总样本数

* 结论：参数更新：
	$$
	\theta_j = \theta_j - \alpha\frac{1}{m}\sum_{i=1}^m(h_\theta(x_i)-y_i)x_i^j
	$$
	其中减号表示下降，$\alpha$为学习率（表示步长的参数）

### 实际操作

* 有了数据后，目标建立一个分类器（求解出参数$θ_0$，$θ_1$，$θ_2$）。需要设定一个阈值（概率值），根据阈值判断分类结果
	```python
	pd_data = pd.read_csv('file.csv')
	```
	
* 需要完成的模块：

	* `sigmoid`：映射到概率的函数
	* `model`：返回预测结果值
	* `cost`：根据参数计算损失
	* `gradient`：计算每个参数的梯度方向
	* `descent`：进行参数更新
	* `accuracy`：计算精度

* `sigmoid`函数：
		
	```python
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))
	```
	
* `model`模块：通过在数据中插入一列（全1），将数值运算转变为矩阵运算：
	$$
	\begin{pmatrix}
	\theta_0 & \theta_1 & \theta_2
	\end{pmatrix}
	\times
	\begin{pmatrix}
	1 \\ x_1 \\ x_2
	\end{pmatrix}
	= \theta_0 + \theta_1x_1 + \theta_2x_2
	$$
	
	```python
	def model(X, theta):
		return sigmoid(np.dot(X, theta.T))  # np.dot()运算矩阵乘法
	pd_data.insert(0, 'Ones', 1)  # 在数据（pd_data）中插入全1列，名为Ones
	orig_data = pd_data.as_matrix()  # 将DataFrame转换为矩阵
	cols = orig_data.shape[1]  # 列数目
	X = orig_data[:, 0:cols-1]  # 一个矩阵，第一列应全为1，后两列分别为x[1], x[2]（这一操作要求特征在前若干列，否则应使用 X = orig_data.ix[:, orig_data.columns != y_label] 获取）
	y = orig_data[:, cols-1:cols]  # 一个矩阵，只有一列，为分类标签（0或1）（这一操作要求分类标签在最后一列，否则应使用 y = orig_data.ix[:, orig_data.columns == y_label] 获取）
	theta = np.zeros((1, 3))  # 用于存放θ参数，先用0填充
	```
	
* 损失函数：将对数似然函数取负号：
	$$
	D(h_\theta(x), y) = -y_i\log(h_\theta(x_i))-(1-y_i)\log(1-h_\theta(x_I))
	$$
	再求平均损失
	$$
	J(\theta) = \frac{\sum_{i=1}^nD(h_\theta(x_i, y_i))}{n}
	$$
	
	```python
	def cost(x, y, theta):
		left = np.multiply(-y, np.log(model(X, theta)))
		right = np.multiply(1-y, np.log(1 - model(X, theta)))
		return np.sum(left - right) / len(X)
	```
	
* 计算梯度：
	$$
	\frac{\part}{\part\theta_j}J = -\frac{\sum_{i=1}^n(y_i-h_\theta(x_j))x_{ij}}{n}
	$$
	
	```python
	def gradient(X, y, theta):
		grad = np.zeros(theta.shape)  # 梯度与θ一一对应
		error = (model(X, theta) - y).ravel()  # ravel()用于转换为一维数组
		for j in range(len(theta.ravel())):  # 每个参数
			term = np.multiply(error, x[:, j])
			grad[0, j] = np.sum(term) / len(X)
		return grad
	```
	
* 停止策略：
	```python
	STOP_ITER = 0
	STOP_COST = 1
	STOP_GRAD = 2
	def stop_crit(type_, value, threshold):  # 三种停止策略
		if type_ == STOP_ITER:  # 根据指定的迭代次数停止迭代（每次更新参数进行计数，达到预定值停止更新）
			return value > threshold
		elif type_ == STOP_COST:  # 目标函数值几乎不再变化，停止迭代
			return abs(value[-1] - value[-2]) < threshold
		elif type_ == STOP_GRAD:  # 梯度几乎不再变化，停止迭代
			return np.linalg.norm(value) < threshold
	```
	
* 迭代过程：
	```python
	def shuffle_data(data):  # 洗牌
		np.random.shuffle(data)
		cols = data.shape[1]
		X = data[:, 0:cols-1]
		y = data[:, cols-1:]
		return X, y

	def descent(data, theta, batch_size, stop_type, thresh, alpha):
		i = 0  # 迭代次数
		k = 0  # batch
		X, y = shuffle_data(data)
		grad = np.zeros(theta.shape)
		costs = [cost(X, y, theta)]

		while True:
			grad = gradient(X[k:k+batch_size], y[k:k+batch_size], theta)
			k += batch_size  # 样本包大小，1表示随机梯度下降，总样本数则表示批量梯度下降，之间则表示小批量
			if k >= n:
				X, y = shuffle_data(data)  # 重新洗牌
				theta -= alpha * grad  # 参数更新，alpha为学习率
				costs.append(cost(X, y, theta))  # 计算新的损失
				i += 1
	
			if stop_type == STOP_ITER:
				value = i
			elif stop_type == STOP_COST:
				value = costs
			elif stop_type == STOP_GRAD:
				value = grad
			if stop_crit(stop_type, value, thresh):  # 判断停止迭代
				break
		return theta, i-1, costs, grad
	```
	
* 预测与精度
	```python
	def predict(X, theta):
		return [1 if x >= 0.5 else 0 for x in model(X, theta)]  # 概率大于0.5认为是1，小于则认为是0，可以自行指定（一般0.5~0.6）
	predictions = predict(X, theta)
	correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
	accuracy = sum(map(int, correct)) % len(correct)
	```

### 一些细节

* 首先应观察样本分布情况：
	
	```python
	count_classes = pd.value_counts(data['class'], sort=True).sort_index()  # `class`为分类指标
	count_classes.plot(kind='bar')
	plt.xlabel('class')
	plt.ylabel('frequency')
	```
	
* 处理样本不均衡：
	
	1. 下采样：减少样本数多的类别，使得两种样本数相当（同样少）
	
	2. 过采样：生成样本数少的类别，使得两种样本数相当（同样多）
	
	```python
	# 下采样，若 y == 1 的样本数较少
	y1_number = len(data[data.y_label == 1])  # 较少的样本数量
	y1_indicies = np.array(data[data.y_label == 1].index)  # 转换成nparray
	y0_indicies = data[data.y_label == 0].index
	random_y0_indicies = np.random.choice(y0_indicies, y1_number, replace=False)  # 从 y==0 的样本中随机选出较少的样本数
	random_y0_indicies = np.array(random_y0_indicies)  # 转换成nparray
	under_sample_indicies = np.concatenate([random_y0_indicies, y1_indicies])  # 结合y==1 和随机选出的 y==0 的样本
	under_sample_data = data.iloc[under_sample_indicies, :]  # 获得样本均衡的data
	X_undersample = under_sample_data.ix[:, under_sample_data.columns != y_label]  # 最后的X
	y_undersample = under_sample_data.ix[:, under_sample_data.columns == y_label]  # 最后的y
	```
	
* 对于不同的特征，机器学习会误认为特征值大的相对重要。因而，当特征的重要程度相当时，需要进行标准化处理
	```python
	from sklearn.preprocessing import StandardScaler
	columns = ['x_1', 'x_2']  # 要进行处理的列标签
	data[columns] = StandardScaler().fit_transform(data[columns])
	```
	
* 交叉验证：
	
	1. 将数据划分为较多的训练集和较少的测试集
	2. 将训练集三等分（具体值可调），用1和2共同建立模型，得到一些参数，再用3（验证集）去验证
	3. 三者轮换，共三次训练+验证
	
	```python
	from sklearn.cross_validation import train_test_split  # 划分训练集和测试集
	# 原始数据集，用于最后的测试
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # test_size表示测试集占据的比例，random_state是随机状态（需要经过洗牌），相同的random_state表示最终得到的划分是一致的
	# 下采样数据集
	X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.3, random_state=0)
	```
	
* 模型评估：对于样本不均衡的案例，精度往往不能反映模型的优劣，因此采用`recall`值（召回率）进行模型评估：
	$$
	recall = \frac{TP}{TP+FN}
	$$
	（`TP`：正类判定为正类；`FP`：负类判定为正类；`FN`：正类判定为负类；`TN`：负类判定为负类）

	```python
	from sklearn.metrics import recall_score
	```
	
* 正则化惩罚：为了防止过拟合（常常由于参数浮动大导致），对$θ$进行正则化处理，使其浮动较小：

	$L_1$正则化：$loss + |w|$；$L_2$正则化：$loss + \frac{1}{2}w^2$

	惩罚力度：$\lambda$，$\lambda$较小（0.01）时惩罚力度小，较大（100）时惩罚力度大，通过交叉验证来评估$\lambda$取何值时效果好

	```python
	from sklearn.linear_model import LogisticRegression
	from sklearn.model_selection import KFold  # 划分训练集

	def printing_kfold_scores(x_train_data, y_train_data):
		fold = KFold(5, shuffle=False)  # 将原始训练集切分为5个，shuffle=False不重新排列
		c_param_range = (0.01, 0.1, 1, 10, 100)  # 正则化惩罚项

		# 可视化
		results_table = pd.DataFrame(index=range(len(c_parameter_range), 2), columns=('c_parameter', 'Mean recall score'))
		results_table['c_parameter'] = c_param_range

		j = 0
		for c_param in c_param_range:  # 在循环中使用不同的c值
			print('-' * 20 + '\nc_parameter: %f\n' % c_param + '-' * 20)  # 显示每一次的c值
	
			recall_accs = []
			for iteration, indices in enumerate(fold, start=1):
				lr = LogisticRegression(C=c_param, penalty='l1')  # 建立逻辑回归模型，选择L_1惩罚
				lr.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())  # 用模型训练
				y_pred_undersample = lr.predict(x_train_data.iloc[indices[1], :].values)  # 进行预测
				recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_undersample)  # 计算召回率
				recall_accs.append(recall_acc)
			print('Iteration %d: recall score = %f' % (iteration, recall_acc))  # 打印每个模型的召回率
	
			results_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)  # 平均召回率
			j += 1
			print('Mean recall score = %f' % np.mean(recall_accs))
		best_c = results_table.loc[results_table['Mean recall score'].idxmax()['c_parameter']]  # 最佳c值
		print('Best c parameter is %f' % best_c)
		return best_c
	
	best_c = printing_kfold_scores(X_Train_undersample, y_train_undersample)  # 调用
	```

* 混淆矩阵：

	|        | Predict 1 | Predict 0 |
	| :----: | :-------: | :-------: |
	| True 1 |    TP     |    FN     |
	| True 0 |    FP     |    TN     |

	`Predict`为预测值，`True`为真值

	绘图：

	```python
	import itertools
	
	def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(classes))
		plt.xticks(tick_marks, classes, rotation=0)
		plt.yticks(tick_marks, classes)
	
		thresh = cm.max()/2
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			plt.text(j, i, cm[i, j], horizontalalignment='center', 
			color='white' if cm[i, j] > thresh else 'black')
	
		plt.tight_layout()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
	```
	
	计算召回率：
	$$
	recall = \frac{TP}{TP+FN}
	$$
	计算精度：
	$$
	accuracy = \frac{TP+TN}{TP+FN+FP+TN}
	$$

* 调节阈值

	```python
	from sklearn.metrics import confusion_matrix
	
	lr = LogisticRegression(C=0.01, penalty = 'l1')  # 假设经过前面的测试，发现best_c = 0.0l
	lr.fit(X_train_undersample, y_train_undersample.values.ravel())  # 用训练集对模型进行训练
	y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)  # 对测试集进行概率值预测
	
	thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # 阈值
	
	plt.figure(figsize=(10, 10))  # 指定画图域
	
	j = 1
	for i in thresholds:
		y_test_predictions_high_recall = y_pred_undersample_proba[:, 1] > i
	
		plt.subplot(3, 3, j)  # 子图位置
		j += 1
	
		cnf_matrix = confusion_matrix(y_test_undersample, y_test_prediction_high_recall)  # 计算混淆矩阵
		np.set_printoptions(precision=2)
	
		print('recall metric in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))  # 打印召回率
	
		class_names = (0, 1)
		plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %d' % i)
	```

* 过采样策略：`SMOTE`算法

	1. 对于少数类中每一个样本$x$，以欧氏距离为标准计算它到少数类样本集中所有样本的距离，得到其K近邻

	2. 根据样本不平衡比例设置一个采样比例，以确定采样倍率$N$，对于每一个少数类样本$x$，从其K近邻中随机选择若干个样本，假设选择的近邻为$x_n$

	3. 对于每一个随机选出的近邻$x_n$，分别于原样本按照如下公式构建新的样本：
		$$
		x_{new} = x + rand(0, 1)\times(\tilde{x} - x)
		$$
		其中$rand(0, 1)$为随机函数，$(\tilde{x}-x)$表示欧氏距离
	
	```python
	from imblearn.over_sampling import SMOTE
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  # 划分训练集与测试集
	oversampler = SMOTE(random_state=0)
	os_X, os_y = oversampler.fit_sample(X_train, y_train)  # 不需要对测试集进行操作
	os_X = pd.DataFrame(os_X)
	os_y = pd.DataFrame(os_y)
	best_c = printing_kfold_scores(os_X, os_y)  # 获取最佳c值
	
	lr = LogisticRegression(C=best_c, penalty='l1')  # 逻辑回归建模
	lr.fit(os_X, os_y.values.ravel())
	y_pred = lr.predict(X_test.values)
	
	cnf_matrix = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
	np.set_printoption(precision=2)
	
	print('recall metric in the testing dataset: ', cnf_matrix[1, 1] / (cnf_matrix[1, 0] + cnf_matrix[1, 1]))  # 打印召回率
	
	class_names = (0, 1)
	plot_confusion_matrix(cnf_matrix, classes=class_names, title='Threshold >= %d' % i)  # 绘制混淆矩阵
	```
	
	过采样往往优于下采样

***

## 决策树算法

### 理论内容

* 决策树：将所有数据从根节点（第一个选择点）开始一步步进行决策分支，最终到达叶子节点（最终的决策结果）

* 可以用来进行分类或回归

* 每个节点相当于在数据中进行了一次划分，得到一个左子树和一个右子树

* 训练决策树：从给定的训练集中构造出一棵树（从根节点开始选择特征，以及如何根据特征进行切分），是任务的难点所在

* 测试决策树：根据构造的树模型走一遍即可

* 如何切分特征（选择节点）：通过一种衡量标准，来计算通过不同特征进行分支选择后的分类情况，找出最好的当成根节点，以此类推

* 衡量标准：熵，表示随机变量不确定性
	$$
	H(x) = - \sum_{i=1}^n P_i\log P_i
	$$
	熵值越高，不确定性越高，即可能的结果越多（即$P\to0.5$）。因此，在分类任务中我们希望熵值极小（即$P\to0$或$P\to1$）

* 信息增益：表示特征$X$使得类$Y$的不确定性减少的程度

	计算：

	1. 未划分时的熵值：
		$$
		H_0(x) = -P(x=0)\log P(x=0) - P(x=1)\log P(x=1)
		$$

	2. 根据某一特征$X$划分，原数据被划分为$m$组，相应的$X$的取值概率分别为$(p_1, p_2, ..., p_m)$

	3. 在上述$m$组中，每一组中$P(x=1) = P_i(x=1)$，则每一组的熵值为：
		$$
		H_i(x) = -P_i(x=0)\log P_i(x=0) - P_i(x=1)\log P_i(x=1)
		$$

	4. 在$X$的划分下，熵值变为：
		$$
		H'(x) = \sum_{i=1}^m p_iH_i(x)
		$$

	5. 因此信息增益为
		$$
		-\Delta H(x) = H_0(x) - H'(x)
		$$

* `ID3`算法：使用信息增益作为衡量标准

	通过遍历，计算每一个划分的信息增益，选取最大的作为根节点，以此类推

	问题：遇到极为稀疏、每组中数据个数极少的特征（此类特征末态熵极小，因而信息增益极大），会将其作为根节点，但这中特征在分类中几乎没有意义（极端例子：`ID值`）

* `C4.5`算法：使用信息增益率作为衡量标准

* `GINI`系数：
	$$
	Gini(p) = \sum_{k=1}^K p_k(1-p_k) = 1 - \sum_{k=1}^K p_k^2
	$$

* `CART`算法：使用`GINI`系数差值作为衡量标准

* 遇到连续值：通过贪婪算法确定分界点

* 决策树剪枝策略：避免过拟合（极端例子：每个叶子节点只有一个数据，无限纯净）

	* 预剪枝：边建立决策树边进行剪枝（简化模型，更实用）

		限制 深度，叶子节点个数，叶子节点样本数，信息增益量 等

	* 后剪枝：建立完决策树以后进行剪枝

		通过一定衡量标准：

		损失 $C(T)$：
		$$
		C(T) = \sum samples \times H(x) \\ or \\ C(T) = \sum samples \times Gini(p)
		$$
		再限制叶子节点个数 $T_{leaf}$：
		$$
		C_\alpha(T) = C(T) + \alpha |T_{leaf}|
		$$
		对每一次划分，计算划分前后的 $C_\alpha$ 值，若划分后变小，则允许进行划分，否则删除这一划分

### 实际操作

* 涉及参数

	```python
	from sklearn import tree
	dtr = tree.DecisionTreeRegressor(max_depth=2)  # 建立树的模型，最大深度为2
	dtr.fit(data.iloc[:, [0, 1]], data.iloc[:, -1].values.ravel())  # 使用data中的前两列作为划分特征，最后一列为分类结果
	```

	在实例化模型时，常用的参数：

	* `max_depth`：最大深度，即只选用最好的几个特征来建立模型（预剪枝）。可以通过循环、交叉验证来获取最合适的深度
	* `min_sample_split`：分裂时所允许的叶子节点中最小的样本数。总样本数较小时无需设定，但若 $>10^5$，可以尝试 $5$

* 可视化显示

	需要先[安装](http://www.graphviz.org/Download..php)工具`graphviz`

	```python
	import pydotplus
	from IPython.display import Image
	
	dot_data = tree.export_graphviz(
		dtr, out_file=None, feature_names=data.columns[0:2], filled=True, impurity=False, rounded=True
		)  # 第一个参数为决策树实例，feature_names需要选择特征对应的列名（此处传入前两列列名）
	
	graph = pydotplus.graph_from_dot_data(dot_data)
	graph.get_nodes()[7].set_fillcolor('#66CCFF')  # 7指所绘的节点个数
	Image(graph.create_png())
	graph.write_png('dtr.png')  # 保存png图片
	```

* 自动选取参数：

	```python
	from sklearn.grid_search import GridSearchCV
	from sklearn.ensemble import RandomForestRegressor  # 随机森林
	from sklearn.model_selection import train_test_split  # 交叉验证
	
	data_train, data_test, target_train, target_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.1, random_state=0)
	
	tree_param_grid = {'min_samples_split': (3, 6, 9), 'n_estimators': (10, 50, 100)}  # RandomForestRegressor 中需要尝试的参数
	grid = GridSearchCV(RandomForestRegressor(), param_grid=tree_param_grid, cv=5)  # 第一个参数为算法，第二个为参数候选项（字典形式），第三个为交叉验证次数
	grid.fit(data_train, target_train)
	print(grid.best_params_)  # 打印最佳参数值
	```

***

## 集成算法与随机森林

* 三种集成算法：

	* _Bagging_：并行地训练多个分类器，取平均：

		$$
		f(x) = \frac{\sum_{m=1}^M f_M(x)}{M}
		$$

		代表：随机森林

	* _Boosting_：串行训练，从弱分类器开始，依次构造树，后一棵树负责弥补前面所有树的残差，从而达到不断加强分类器的目的

		$$
		F_m(x) = F_{m-1}(x) + argmin_h \sum_{i=1}^m L(y_i, F_{m-1}(x_i)+h(x_i))
		$$
	
		代表：_AdaBoost_，_XgBoost_
	
	* _Stacking_：堆叠，聚合多个分类或回归模型
	
		第一阶段训练多个分类器（不同算法），得到一系列分类结果；第二阶段将之前得到的分类结果作为特征输入到一个新的分类器中，得到最终结果

### 随机森林

* 随机森林：每个决策树数据采样和特征选择都是随机的，最后将很多个决策树并行放在一起，进行最终的决策

* 实现随机：每棵树随机采样（通常选择原始数据的$60\%$至$80\%$），再从全部特征中随机选取若干个。但要保证每棵树采样数相等、特征数相等

* 随机森林的优势：便于观测哪些特征比较重要，方法如下：

	例如，要观测A特征的重要性，就在拥有全部特征的情况下获得结果的误差值$err_1$，再将A特征完全破坏掉（用随机噪音替换），获得结果的误差值$err_2$，若$err_1 \approx err_2$，则说明A特征不重要；反之若$err_1 \ll err_2$，则说明A特征很重要

* 理论上树的数量越多，泛化能力越强，但实际上树的个数较多时，会趋于一个稳定值

### AdaBoost

* 流程：先对所有数据赋以相同的权重，然后进行分类。如果某一数据在分类中分错了，就再下一次赋以更大的权重。如此构造多个分类器，每个分类器具有不同的精度；于是根据每个分类器的准确性来确定各自的权重，得到最后的结果

***

## 案例实战——泰坦尼克号获救预测

### 数据预处理

* 观察数据：

	```python
	import seaborn as sns
	
	titanic = sns.load_dataset('titanic.csv')
	print(titanic.describe(include='all'))
	```

	得到的结果如下：

	||survived|pclass|sex|age|sibsp|parch|fare|embarked|class|who|adult_male|deck|embark_town|alive|alone|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|count|891.000000|891.000000|891|714.000000|891.000000|891.000000|891.000000|889|891|891|891|203|889|891|891|
|unique|NaN|NaN|2|NaN|NaN|NaN|NaN|3|3|3|2|7|3|2|2|
|top|NaN|NaN|male|NaN|NaN|NaN|NaN|S|Third|man|True|C|Southampton|no|True|
|freq|NaN|NaN|577|NaN|NaN|NaN|NaN|644|491|537|537|59|644|549|537|
|mean|0.383838|2.308642|NaN|29.699118|0.523008|0.381594|32.204208|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|std|0.486592|0.836071|NaN|14.526497|1.102743|0.806057|49.693429|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|min|0.000000|1.000000|NaN|0.420000|0.000000|0.000000|0.000000|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|25%|0.000000|2.000000|NaN|20.125000|0.000000|0.000000|7.910400|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|50%|0.000000|3.000000|NaN|28.000000|0.000000|0.000000|14.454200|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|75%|1.000000|3.000000|NaN|38.000000|1.000000|0.000000|31.000000|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|
|max|1.000000|3.000000|NaN|80.000000|8.000000|6.000000|512.329200|NaN|NaN|NaN|NaN|NaN|NaN|NaN|NaN|

* 分析：

	* 共计891个样本，其中`age`有一定缺失，但是年龄对于是否获救有较大影响，不应舍弃，故进行缺失值处理，用均值填充。

	```python
	titanic['age'] = titanic['age'].fillna(titanic['age'].median())
	```

	* `sex`在数据集中用`male`和`female`表述，不便于后期处理，故改为0和1

	```python
	titanic.loc[titanic['sex'] == 'male', 'sex'] = 0
	titanic.loc[titanic['sex'] == 'female', 'sex'] = 1
	```

	* 同样的，可以对上船地点`embarked`进行数值映射，但上船地点不像性别那样，显而易见只有两种而可能，所以需要先查看其所有取值可能：

	```python
	print(titanic['embarked'].unique())
	```

	然后进行数值映射：

	```python
	titanic['embarked'] = titanic['embarked'].fillna('S')  # 由于只有两个缺失值，故使用最多的`S`填充
	titanic.loc[titanic['embarked'] == 'S', 'embarked'] = 0
	titanic.loc[titanic['embarked'] == 'C', 'embarked'] = 1
	titanic.loc[titanic['embarked'] == 'Q', 'embarked'] = 2
	```
  
	* 标准化处理：

	```python
	predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
	titanic[predictors] = pd.DataFrame(StandardScaler().fit_transform(titanic[predictors]))
	```

* 在下面的操作中，将`survived`作为分类标签，`pclass`、`sex`、`age`、`sibsp`、`parch`、`fare`、`embarked`为分类特征。（其余参数不是与这些特征重复，如`class`、`who`等，就是缺失值太多，如`deck`，故不作为特征）

### 线性回归预测

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
predictions = list()

alg = LinearRegression()
kf = KFold(4, random_state=False)  # 切分成4份

for train, test in kf.split(titanic):
	train_predictors = titanic[predictors].iloc[train, :]  # 训练数据
	train_target = titanic['survived'].iloc[train]  # 训练集标签
	alg.fit(train_predictors, train_target)  # 代入线性回归模型
	test_predictions = alg.predict(titanic[predictors].iloc[test, :])  # 预测测试集
	predictions.append(test_predictions)  # 记录测试结果

predictions = np.concatenate(predictions, axis=0)  # 
predictions[predictions > 0.5] = 1  # 将预测值映射为0或1的结果
predictions[predictions <= 0.5] = 0

accuracy = predictions[predictions == titanic['survived']].shape[0] / len(predictions)  # 预测精度
print(accuracy)
```

> 运行结果：0.7890011223344556，对于二分类任务来说比较低

### 逻辑回归预测

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
predictions = list()

alg = LogisticRegression(random_state=1, solver='lbfgs', penalty='l2')
kf = KFold(4, random_state=False)  # 切分成4份

for train, test in kf.split(titanic):
	train_predictors = titanic[predictors].iloc[train, :]  # 训练数据
	train_target = titanic['survived'].iloc[train]  # 训练集标签
	alg.fit(train_predictors, train_target)  # 代入线性回归模型
	test_predictions = alg.predict(titanic[predictors].iloc[test, :])  # 预测测试集
	predictions.append(test_predictions)  # 记录测试结果

predictions = np.concatenate(predictions, axis=0)  # 
predictions[predictions > 0.5] = 1  # 将预测值映射为0或1的结果
predictions[predictions <= 0.5] = 0

accuracy = predictions[predictions == titanic['survived']].shape[0] / len(predictions)  # 预测精度
print(accuracy)
```

> 运行结果：0.7934904601571269

* 如果只是要评估精度，可以直接用`cross_val_score`代替：

	```python
	from sklearn.linear_model import LogisticRegression
	from sklearn. model_selection import cross_val_score
	
	predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
	alg = LogisticRegression(random_state=1, solver='lbfgs', penalty='l2')
	scores = cross_val_score(alg, titanic[predictors], titanic['survived'], cv=5)
	print(scores.mean())
	```

### 随机森林预测

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
predictions = []

alg = RandomForestClassifier(random_state=1, criterion='gini', n_estimators=50, min_samples_split=4, min_samples_leaf=2)
# n_estimators表示构造的树的个数，min_samples_split表示最小切分样本数（什么时候停止切割），min_samples_leaf表示最小叶子节点个数
kf = KFold(4, random_state=False)

for train, test in kf.split(titanic):
	train_predictors = titanic[predictors].iloc[train, :]  # 训练数据
	train_target = titanic['survived'].iloc[train]  # 训练集标签
	alg.fit(train_predictors, train_target)  # 代入线性回归模型
	test_predictions = alg.predict(titanic[predictors].iloc[test, :])  # 预测测试集
	predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0

accuracy = predictions[predictions == titanic['survived']].shape[0] / len(predictions)
print(accuracy)
```

> 运行结果：0.8338945005611672

* 特别注意：随机森林在使用时需要对参数（树的个数，最小切分样本数等）进行调优

* 评估特征重要性：

	```python
	predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
	
	selector = SelectKBest(f_classif, k=5)
	selector.fit(titanic[predictors], titanic['survived'])
	
	scores = -np.log10(selector.pvalues_)
	
	plt.bar(range(len(predictors)), scores)
	plt.xticks(range(len(predictors)), predictors, rotation=60)
	plt.show()
	
	for index, feature in enumerate(predictors):
		print('%s:\t%d' % (feature, scores[index]))
	```

	> 运行结果：
	> | feature  | score |
	> | -------- | ----- |
	> | pclass   | 24    |
	> | sex      | 68    |
	> | age      | 1     |
	> | sibsp    | 0     |
	> | parch    | 1     |
	> | fare     | 14    |
	> | embarked | 2     |
	>

### 集成算法预测

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
predictions = []

algs = [
	[GradientBoostingClassifier(random_state=1, n_estimators=40, max_depth=3), predictors],
	[LogisticRegression(random_state=1, solver='lbfgs'), predictors]
]  # 包含两种算法
kf = KFold(4, random_state=False)

for train, test in kf.split(titanic):
	train_predictors = titanic[predictors].iloc[train, :]  # 训练数据
	train_target = titanic['survived'].iloc[train]  # 训练集标签
	full_test_predictions = []  # 两种算法总体的预测

	for alg, predictor in algs:  # 对每种算法进行拟合
		alg.fit(train_predictors, train_target)  # 代入线性回归模型
		test_predictions = alg.predict(titanic[predictors].iloc[test, :])  # 预测测试集
		full_test_predictions.append(test_predictions)

	test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2  # 结果取两种算法的平均
	test_predictions[test_predictions > 0.5] = 1
	test_predictions[test_predictions <= 0.5] = 0
	predictions.append(test_predictions)  # 真正的预测结果

predictions = np.concatenate(predictions, axis=0)
accuracy = predictions[predictions == titanic['survived']].shape[0] / len(predictions)
print(accuracy)
```

> 运行结果：0.8114478114478114

***

## 贝叶斯算法

### 理论内容

* 贝叶斯要解决的问题：
	* 正向概率：知道可能事件的分布（如袋子中每种颜色小球的数量），求每个事件发生的概率（摸到某种颜色小球的概率）
	* 逆向概率：事先不知道事件分布的比例（现实中大部分事件都是这样的，我们只能观测到事物表面），而是通过对每个事件发生概率的观测（试验），反推出事件是如何分布的

* 贝叶斯公式：
	$$
	P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
	$$
	用于：对于观测数据$D$的每种猜测$h_i$，有：
	$$
	P(h_i|D) = \frac{P(h_i) \cdot P(D|h_i)}{P(D)} \propto P(h_i) \cdot P(D|h_i)
	$$
	其中第一项$P(h)$称为先验概率，即这种猜测在以往的数据中发生的概率，而第二项$P(D|h)$表示“猜测生成观测数据的可能性的大小”
	
* 模型比较理论
	
	* 最大似然：最符合观测数据的（即$P(D|h)$最大的）最有优势
	* 奥卡姆剃刀：最常见的（即$P(h)$最大的）最有优势
		例：进行数据拟合时，越是高阶的多项式越不常见
	
* 朴素贝叶斯问题：

	对于一个数据$D$由特征$d_1, d_2, ..., d_n$组成的问题：
	$$
	P(D|h) = P(d_1, d_2, ..., d_n|h) = P(d_1|h) \cdot P(d_2|d_1, h) \cdot ... \cdot P(d_n|d_1, ..., d_{n-1}, h)
	$$
	进行假设：特征$d_i$之间互相独立，则简化为：
	$$
	P(D|h) = P(d_1|h) \cdot P(d_2|h) \cdot ... \cdot P(d_n|h)
	$$
  

### 案例：拼写纠正

* $P(h)$在本例中表示猜测词在全文本中出现的词频，而$P(D|h)$可以用编辑距离或是两个字母在键盘上的距离来衡量。枚举所有可能的$h$并选取概率最大的

* 训练模型：根据[语料库](<https://corpus.byu.edu/nowtext-samples/text.zip>)中每个词的词频进行评测（$P(h)$）

	```python
	from re import findall
	from collections import defaultdict
	
	def words(text):  # 只保留单词
		return findall('[a-z]+', text.lower())
	
	def train(features):
		model = defaultdict(lambda: 1)  # 确保每个词（不限于语料库中的）至少出现一次，以防专有词汇等
		for feature in features:
			model[feature] += 1
		return model
	
	WORDS = train(words(open('text.txt').read()))  # 用语料库进行训练
	```

* 编辑距离：$P(D|h)$

	```python
	from string import ascii_lowercase  # 小写字母表
	
	def edit_d(word: str, d: int = 1) -> set:
		"""
		find all the words whose edit distance to `word` is `d`
		:param word: typo word
		:param d: edit distance
		:return: a set of filtered words
		"""
		n = len(word)
		if d == 1:  # 返回编辑距离为1的所有字母组合
			return set(
				[word[0:i] + word[i+1:] for i in range(n)] +   # 删除
				[word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(n-1)] +   # 易位
				[word[0:i] + c + word[i+1:] for i in range(n) for c in ascii_lowercase] +   # 替换
				[word[0:i] + c + word[i:] for i in range(n+1) for c in ascii_lowercase]  # 插入
			)
		if d == 2:  # 返回编辑距离为2的所有字母组合
			return set(e2 for e1 in edit_d(word, d=1) for e2 in edit_d(e1, d=1))
		else:
		raise Exception('The edit distance `d` can only be 1 or 2.')
	```

* 最终

	```python
	def known(words_: [list, set]) -> set:  # 判断是否在语料库内
		"""
		return the correct words in `words_`
		:param words_: a set of words
		:return: correct words
		"""
		return set(w for w in words_ if w in WORDS)
	
	def correct(word):
		word = word.lower()
		candidates = known([word]) or known(edit_d(word, d=1)) or known(edit_d(word, d=2)) or [word]  # 依次为原词（已知）、编辑距离1的词、编辑距离2的词、原词（未知）
		return max(candidates, key=lambda w: WORDS[w])
	```

* 测试运行：

	```python
	print(correct('worls'))
	```
	运行结果：
	
	> world

***

## 文本分析

* 分词：将中文句子切分为单个词汇

	使用 _结巴分词_ 模块：

	```python
	import jieba
	```

* 停用词：在语料中大量出现，对于主题分析没有任何实际意义的词汇（如：标点符号，“你”、“一般”等）。故在进行文本分析时应最先删除停用词

* 关键词提取：`TF-IDF`

	* `TF`：词频统计
		$$
		TF = \frac{某个词在文章中出现的次数}{文章中总词数}
		$$

	* `IDF`：逆文档频率
		$$
		IDF = \log \frac{语料库的文档总数}{包含该词的文档数+1}
		$$
		如果某个词很少见，但在文章中多次出现，那它就很可能反映了这篇文章的特征

	* `TF-IDF`评估方法：
		$$
		TF\_IDF = TF \times IDF
		$$
		值越高，越有可能是我们所需要的特征词汇

* 相似度：将两个句子转化为词频向量，再根据这两个向量进行相似度分析（余弦相似度）
	$$
	\begin{align}
	\cos \theta &= \frac{\sum_{i=1}^n A_i \times B_i}{\sqrt{\sum_{i=1}^n (A_i)^2} \times \sqrt{\sum_{i=1}^n (B_i)^2}} \\
	&= \frac{A \cdot B}{|A| \times |B|}
	\end{align}
	$$
	
***

## 支持向量机

