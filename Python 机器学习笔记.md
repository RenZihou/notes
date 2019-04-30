# Python 机器学习笔记

***

## Numpy库的使用：
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

## Pandas库的使用：
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

* 自定义函数：`df.apply(user_func)` 

	例：返回每一列非空值的个数

	```python
	df.apply(lambda x: len(x[pd.isnull(x) == False]))
	```

* `DataFrame`内部结构是`Series`，而`Series`内部结构是`ndarray`

***

## Matplotlib库的使用：
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

## Seaborn库的使用：
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

## 逻辑回归：二分类算法

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

* 处理样本不均衡：法一：下采样：减少样本数多的类别，使得两种样本数相当（同样少）；法二：过采样：生成样本数少的类别，使得两种样本数相当（同样多）
	
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
	data['norm_x1'] = StandardScaler().fit_transform(data['x1'].reshape(-1, 1))  # 对x1进行标准化预处理，其中reshape()中的-1表示自动识别得到目标矩阵行数，1表示目标矩阵列数
	```
	
* 交叉验证：首先，数据会被划分为较多的训练集和较少的测试集；再将训练集三等分，用1和2共同建立`model`，得到一些参数，再用3（验证集）去验证；此后三者轮换，共三次训练+验证。目的是让验证更加可靠
	```python
	from sklearn.cross_validation import train_test_split
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
	from sklearn.linear_model import LogisticRegression
	from sklearn.cross_validation import KFold, cross_vval_score
	from sklearn.metrics import confusion_matrix, recall_score, classification_report
	```
	
* 正则化惩罚：为了防止过拟合，对$θ$进行正则化处理，使其浮动较小：

   $L_1$正则化：$loss + |w|$；$L_2$正则化：$loss + \frac{w^2}{2}$

   惩罚力度：$\lambda L_2$，$λ$较小（0.01）时惩罚力度小，较大（100）时惩罚力度大