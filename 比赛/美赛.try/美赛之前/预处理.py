# coding=utf-8
from __future__ import division
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import trange
import time
import re


def classify(input_vct, data_set):
    data_set_size = data_set.shape[0]
    diff_mat = np.tile(input_vct, (data_set_size, 1)) - \
               data_set  # 扩充input_vct到与data_set同型并相减
    sq_diff_mat = diff_mat ** 2  # 矩阵中每个元素都平方
    distance = sq_diff_mat.sum(axis=1) ** 0.5  # 每行相加求和并开平方根
    return distance.min(axis=0)  # 返回最小距离


def LDA(X, y):
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 2])

    len1 = len(X1)
    len2 = len(X2)

    mju1 = np.mean(X1, axis=0)  # 求中心点
    mju2 = np.mean(X2, axis=0)

    cov1 = np.dot((X1 - mju1).T, (X1 - mju1))
    cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
    Sw = cov1 + cov2

    w = np.dot(np.linalg.pinv(Sw),
               (mju1 - mju2).reshape((len(mju1), 1)))  # 计算w
    X1_new = np.dot(X1, w)
    X2_new = np.dot(X2, w)
    y1_new = [1 for i in range(len1)]
    y2_new = [2 for i in range(len2)]

    X_new = np.concatenate((X1_new, X2_new), axis=0)
    y_new = np.concatenate((y1_new, y2_new), axis=0)

    return X_new, y_new


def normal_or_not(x):  # 辅助函数
    #     if (re.match(x,'normal')) is None:
    if x == 'normal.':
        return 1
    else:
        return 2


def preprocess(file_name, tag):
    df = pd.read_csv(file_name,header=None)  # 查看pandas官方文档发现，read_csv读取时会自动识别表头，数据有表头时不能设置header为空（默认读取第一行，即header=0)；数据无表头时，若不设置header，第一行数据会被视为表头，应传入names参数设置表头名称或设置header=None。

    df.loc[:,41] = df.loc[:,41].apply(lambda x: normal_or_not(x))  # 转换后提取最后一列
    class_label = np.array(df.loc[:,41])
    print(class_label)
    data = df.drop(columns=41)

    data = (pd.get_dummies(data))  # onehot编码，会自动将是字符串的列转为onehot编码，如果列名是数字可能不会再扩充充，而是把那一列转为数字型
    np.savetxt(str(tag) + 'onehot2.csv', data, delimiter=',')
    print(file_name + '完成nominal数据转换')

    # data, class_label = LDA(data, class_label)
    # print(file_name+'完成数据降维')

    '''
    model = SMOTEENN(random_state=0)
    data, class_label = model.fit_sample(data, class_label)
    print(file_name+'完成上下采样处理')
    '''
    model = SMOTE()  # 采样
    data, class_label = model.fit_sample(data, class_label)
    print(file_name + '完成上下采样处理')

    mm = MinMaxScaler()
    data = mm.fit_transform(data)
    print(file_name + '完成数据归一化')

    np.savetxt(str(tag) + 'data.csv', data, delimiter=',')
    np.savetxt(str(tag) + 'label.csv', class_label, delimiter=',')

    return data, class_label


def roc(data_set):
    normal = 0
    data_set_size = data_set.shape[1]
    roc_rate = np.zeros((2, data_set_size))
    for i in trange(data_set_size):
        if data_set[2][i] == 1:  # 2代表？
            normal += 1
    abnormal = data_set_size - normal
    max_dis = data_set[1].max()
    for j in range(1000):
        threshold = max_dis / 1000 * j
        normal1 = 0
        abnormal1 = 0
        for k in range(data_set_size):
            if data_set[1][k] > threshold and data_set[2][k] == 1:
                normal1 += 1
            if data_set[1][k] > threshold and data_set[2][k] == 2:
                abnormal1 += 1
        roc_rate[0][j] = normal1 / normal  # 阈值以上正常点/全体正常的点
        roc_rate[1][j] = abnormal1 / abnormal  # 阈值以上异常点/全体异常点
    return roc_rate


def test(training_filename, test_filename):
    start = time.time()
    training_mat, training_label = preprocess(training_filename, 'train_')
    end = time.time()

    print('训练集预处理完毕,用时', end - start, '秒')
    start = time.time()
    test_mat, test_label = preprocess(test_filename, 'test_')
    end = time.time()
    print('测试集预处理完毕,用时', end - start, '秒')

    '''
    training_mat=np.array(pd.read_csv('1data.csv',header=None))
    training_label=np.array(pd.read_csv('1label.csv',header=None))
    test_mat=np.array(pd.read_csv('2data.csv',header=None))
    test_label=np.array(pd.read_csv('2label.csv',header=None))

    test_size = test_mat.shape[0]
    result = np.zeros((test_size, 3))
    for i in trange(test_size):
        # 序号， 最小欧氏距离， 测试集数据类别
       result[i] = i + 1, classify(test_mat[i], training_mat), test_label[i]
    print('欧氏距离计算完毕，开始作图')
    result = np.transpose(result)  # 矩阵转置
    plt.figure(1)
    plt.scatter(result[0], result[1], c=result[2],
                edgecolors='None', s=1, alpha=1)
    # 图1 散点图：横轴为序号，纵轴为最小欧氏距离，点中心颜色根据测试集数据类别而定， 点外围无颜色，点大小为最小1，灰度为最大1
    roc_rate = roc(result)
    plt.figure(2)
    plt.scatter(roc_rate[0], roc_rate[1], edgecolors='None', s=1, alpha=1)
    # 图2 ROC曲线， 横轴误报率，即阈值以上正常点/全体正常的点；纵轴检测率，即阈值以上异常点/全体异常点
    plt.show()
    '''


if __name__ == "__main__":
    test('kddcup.data_10_percent_corrected.csv', 'kddcup.data.corrected.csv')  # 直接右击重命名加后缀不容易出现编码问题
