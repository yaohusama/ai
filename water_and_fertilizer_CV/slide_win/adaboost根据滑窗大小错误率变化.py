# -*-coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-10-10
"""
'''
    numFeat = len((open(fileName).readline().split('\n')))
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split(',')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat
'''

def loadDataSet(fileName):

    data=pd.read_csv(fileName)
    labelMat=list(data['tick'])
    data=data.drop(['tick','id'],axis=1)
    # data.index=None
    # dataMat=np.array(data.values)
    # print(data)
    dataMat=data.values.tolist()

    # print(dataMat)
    return dataMat,labelMat

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树分类函数
    Parameters:
        dataMatrix - 数据矩阵
        dimen - 第dimen列，也就是第几个特征
        threshVal - 阈值
        threshIneq - 标志
    Returns:
        retArray - 分类结果
    """
    retArray = np.ones((np.shape(dataMatrix)[0], 1))  # 初始化retArray为1
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0  # 如果小于阈值,则赋值为-1
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0  # 如果大于阈值,则赋值为-1
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳的单层决策树
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        D - 样本权重
    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClasEst - 最佳的分类结果
    """
    dataMatrix = np.mat(dataArr);
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 100.0;
    bestStump = {};
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = float('inf')  # 最小误差初始化为正无穷大
    for i in range(n):  # 遍历所有特征
        rangeMin = dataMatrix[:, i].min();
        rangeMax = dataMatrix[:, i].max()  # 找到特征中最小的值和最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:  # 大于和小于的情况，均遍历。lt:less than，gt:greater than
                threshVal = (rangeMin + float(j) * stepSize)  # 计算阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)  # 计算分类结果
                errArr = np.mat(np.ones((m, 1)))  # 初始化误差矩阵
                errArr[predictedVals == labelMat] = 0  # 分类正确的,赋值为0
                weightedError = D.T * errArr  # 计算误差
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:  # 找到误差最小的分类方式
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    使用AdaBoost算法提升弱分类器性能
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        numIt - 最大迭代次数
    Returns:
        weakClassArr - 训练好的分类器
        aggClassEst - 类别估计累计值
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)  # 初始化权重
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        # print(i)
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  # 构建单层决策树
        # print("D:",D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))  # 计算弱学习算法权重alpha,使error不等于0,因为分母不能为0
        bestStump['alpha'] = alpha  # 存储弱学习算法权重
        weakClassArr.append(bestStump)  # 存储单层决策树
        # print("classEst: ", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)  # 计算e的指数项
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()  # 根据样本权重公式，更新样本权重
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        aggClassEst += alpha * classEst  # 计算类别估计累计值
        # print("aggClassEst: ", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))  # 计算误差
        errorRate = aggErrors.sum() / m
        # print("total error: ", errorRate)
        if errorRate == 0.0:
            print('success')
            break  # 误差为0，退出循环
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    Parameters:
        datToClass - 待分类样例
        classifierArr - 训练好的分类器
    Returns:
        分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):  # 遍历所有分类器，进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)
# -*-coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
"""
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Zhihu:
    https://www.zhihu.com/people/Jack--Cui/
Modify:
    2017-10-11
"""
'''if __name__ == '__main__':
    dataArr, classLabels = loadDataSet('res1.csv')

    # testArr, testLabelArr = loadDataSet('res1.csv')
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    import numpy as np
    from sklearn.model_selection import train_test_split
    # X, y = np.arange(10).reshape((5, 2)), range(5)

    X = np.array(dataArr)
    y = np.array(classLabels)
    # skf = StratifiedKFold(n_splits=2)
    # for train_index, test_index in skf.split(X, y):
    #     print("TRAIN:", train_index, "TEST:", test_index)
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)#random_state=42
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME.R", n_estimators = 10)
    # print(y_test)
    # bdt.fit(dataArr, classLabels)
    bdt.fit(X_train,y_train)
    predictions = bdt.predict(X_train)
    errArr = np.mat(np.ones((len(X_train), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != y_train].sum() / len(X_train) * 100))
    predictions = bdt.predict(X_test)
    errArr = np.mat(np.ones((len(X_test), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != y_test].sum() / len(X_test) * 100))'''

if __name__ == '__main__':
    # dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    err_list={}
    for i in range(1, 20):
        outputfile = 'pcares_winsize' + str(i) + '.csv'
        print(outputfile)
        dataArr, LabelArr = loadDataSet(outputfile)
        from sklearn.model_selection import train_test_split
        X = np.array(dataArr)
        y = np.array(LabelArr)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # random_state=42
        weakClassArr, aggClassEst = adaBoostTrainDS(X_train,y_train)
        # testArr, testLabelArr = loadDataSet('res.csv')
        # print(weakClassArr)
        predictions = adaClassify(X_train, weakClassArr)
        errArr = np.mat(np.ones((len(X_train), 1)))
        err1=float(errArr[predictions != np.mat(y_train).T].sum() / len(X_train) * 100)
        print('训练集的错误率:%.3f%%' % err1)
        predictions = adaClassify(X_test, weakClassArr)
        errArr = np.mat(np.ones((len(X_test), 1)))
        err2=float(errArr[predictions != np.mat(y_test).T].sum() / len(X_test) * 100)
        print('测试集的错误率:%.3f%%' % err2)
        err_list[i]=err2
    err_list=pd.DataFrame([err_list]) #字典转为dataframe内部需要加个[]
    err_list.to_csv('根据滑窗大小变化的测试集错误率.csv')
        