import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

titanic = sns.load_dataset('titanic')
# print(titanic.head())
# print(titanic.describe(include='all'))

titanic['age'] = titanic['age'].fillna(titanic['age'].median())
titanic.loc[titanic['sex'] == 'male', 'sex'] = 0
titanic.loc[titanic['sex'] == 'female', 'sex'] = 1

# print(titanic['embarked'].unique())
titanic['embarked'] = titanic['embarked'].fillna('S')  # 由于只有两个缺失值，故使用最多的`S`填充
titanic.loc[titanic['embarked'] == 'S', 'embarked'] = -1
titanic.loc[titanic['embarked'] == 'C', 'embarked'] = 0
titanic.loc[titanic['embarked'] == 'Q', 'embarked'] = 1

titanic[['age', 'embarked', 'sex', 'fare', 'sibsp', 'parch', 'pclass']] = pd.DataFrame(
    StandardScaler().fit_transform(titanic[['age', 'embarked', 'sex', 'fare', 'sibsp', 'parch', 'pclass']]))
# print(titanic.describe(include='all'))


def linear():
    """
    线性回归预测
    :return:
    """
    predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    predictions = list()
    alg = LinearRegression()
    kf = KFold(4, random_state=False)

    for train, test in kf.split(titanic):
        train_predictors = titanic[predictors].iloc[train, :]  # 训练数据
        train_target = titanic['survived'].iloc[train]  # 训练集标签
        alg.fit(train_predictors, train_target)  # 代入线性回归模型
        test_predictions = alg.predict(titanic[predictors].iloc[test, :])  # 预测测试集
        predictions.append(test_predictions)

    predictions = np.concatenate(predictions, axis=0)
    predictions[predictions > 0.55] = 1
    predictions[predictions <= 0.55] = 0

    accuracy = predictions[predictions == titanic['survived']].shape[0] / len(predictions)
    print(accuracy)
    return alg


def logistic():
    """
    逻辑回归预测
    :return:
    """
    predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    predictions = []
    alg = LogisticRegression(random_state=1, solver='lbfgs', penalty='l2')
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
    # scores = cross_val_score(alg, titanic[predictors], titanic['survived'], cv=5)
    # print(scores.mean())
    return alg


def forest():
    """
    随机森林预测
    :return:
    """
    predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    predictions = []
    alg = RandomForestClassifier(
        random_state=1, criterion='gini', n_estimators=50, min_samples_split=4, min_samples_leaf=2)
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
    # scores = cross_val_score(alg, titanic[predictors], titanic['survived'], cv=5)
    # print(scores.mean())
    return alg


def evaluate():
    """
    评估参数重要程度
    :return:
    """
    predictors = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

    selector = SelectKBest(f_classif, k=5)
    selector.fit(titanic[predictors], titanic['survived'])

    scores = -np.log10(selector.pvalues_)

    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation=60)
    plt.show()

    for index, feature in enumerate(predictors):
        print('%s:\t%d' % (feature, scores[index]))
    return None


def boost():
    """
    集成算法预测
    :return:
    """
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
    return None


if __name__ == '__main__':
    linear()
    logistic()
    forest()
    evaluate()
    boost()
    pass
