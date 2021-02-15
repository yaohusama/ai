# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
# from collections import Counter
#
influence_data=pd.read_csv('influence_data.csv')
# c=Counter(influence_data['influencer_main_genre'])#计算各个出现量的数量
# tmp=influence_data[['influencer_id','follower_id']]
# print(type(tmp.iloc[0]))
# print(tmp[0]['influencer_id'])
min_val=0x7fffffff
max_val=0
print(len(influence_data['influencer_id']))#42770条数据
total_num=set()
unique_num=set()
max_val=5603#42770条数据
mat=[[0] * max_val for i in range(max_val)]
for tmp in zip(influence_data['influencer_id'],influence_data['follower_id']):#还没筛选空值，不过从结果 来看没有空值
    total_num.add(tmp[0])
    total_num.add(tmp[1])
    tmp1=min(tmp[0],tmp[1])
    min_val=min(min_val,tmp1)
    tmp1=max(tmp[0],tmp[1])
    max_val=max(max_val,tmp1)
total_num=list(total_num)
for tmp in zip(influence_data['influencer_id'], influence_data['follower_id']):
    x=total_num.index(tmp[0])
    y=total_num.index(tmp[1])
    mat[x][y]=1
for i in mat:
    flag = False
    for j in i:
        if j == 1:
            flag = True
            break
if flag:
    print("yes")
    # print('fuck')
# mat1=pd.DataFrame(mat)
# mat1.to_csv('出入关系矩阵.csv')
    # print(tmp[0],tmp[1])
# print(min_val,max_val)#74 3670556不从1 开始，考虑其他文件可能有从1开始的其他人id，就不重新编号了
# max_val=42770#42770条数据
# mat=[[1] * max_val for i in range(max_val)]
total_num_len=len(total_num)#5603个不同的记录
print(total_num_len)
# print(total_num)
# mat= [[ 0,  0, 0, 0, 1.],
#  [ 1,  0,  0,  0,  0.],
#  [ 1,  0,  0,  0,  0.],
#  [ 1,  1,  0,  0,  0.],
#  [ 0,  1,  1,  1, 0.]]
import numpy as np
def divide_util(sum,L):
    li=[]
    for i in zip(sum,L):
        li.append((float)(i[0]/i[1]))
    return np.array(li)

class CPageRank(object):
    '''实现PageRank Alogrithm
    '''

    def __init__(self):
        self.PR = []  # PageRank值

    def GetPR(self, IOS, alpha, max_itrs, min_delta):
        '''幂迭代方法求PR值
        :param IOS       表示网页出链入链关系的矩阵,是一个左出链矩阵
        :param alpha     阻尼系数α，一般alpha取值0.85
        :param max_itrs  最大迭代次数
        :param min_delta 停止迭代的阈值
        '''
        # IOS左出链矩阵, a阻尼系数alpha, N网页总数
        N = np.shape(IOS)[0]
        # 所有分量都为1的列向量
        e = np.ones(shape=(N, 1))
        # 计算网页出链个数统计
        L = [np.count_nonzero(e) for e in IOS.T]
        # print(l)
        # print('------------------------------')
        # 计算网页PR贡献矩阵helpS，是一个左贡献矩阵
        # if(l!=0)
        # helps_efunc = lambda ios, l: ios / l
        # helps_func = np.frompyfunc(helps_efunc, 2, 1)
        # helpS = helps_func(IOS, L)
        helpS=divide_util(IOS,L)
        # P[n+1] = AP[n]中的矩阵A
        A = alpha * helpS + ((1 - alpha) / N) * np.dot(e, e.T)
        print('左出链矩阵:\n', IOS)
        print('左PR值贡献概率矩阵:\n', helpS)
        # 幂迭代法求PR值
        for i in range(max_itrs):
            if 0 == np.shape(self.PR)[0]:  # 使用1.0/N初始化PR值表
                self.PR = np.full(shape=(N, 1), fill_value=1.0 / N)
                print('初始化的PR值表:', self.PR)
            # 使用PR[n+1] = APR[n]递推公式，求PR[n+1]
            old_PR = self.PR
            self.PR = np.dot(A, self.PR)
            # 如果所有网页PR值的前后误差 都小于 自定义的误差阈值，则停止迭代
            D = np.array([old - new for old, new in zip(old_PR, self.PR)])
            ret = [e < min_delta for e in D]
            if ret.count(True) == N:
                print('迭代次数:%d, succeed PR:\n' % (i + 1), self.PR)
                break
        return self.PR


def CPageRank_manual():
    # 表示网页之间的出入链的关系矩阵，是一个左关系矩阵，可以理解成右入链矩阵
    # IOS[i, j]表示网页j对网页i有出链
    IOS = np.array(mat, dtype=float)
    pg = CPageRank()
    ret = pg.GetPR(IOS, alpha=0.85, max_itrs=100, min_delta=0.0001)
    res=pd.DataFrame(ret)
    res.to_csv('幂迭代法最后PR值.csv')
    print('最终的PR值:\n', ret)


if __name__ == '__main__':
    CPageRank_manual()