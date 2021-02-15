# -*- coding: utf-8 -*-
from numpy import *
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

    # print(tmp[0],tmp[1])
# print(min_val,max_val)#74 3670556不从1 开始，考虑其他文件可能有从1开始的其他人id，就不重新编号了
# max_val=42770#42770条数据
# mat=[[1] * max_val for i in range(max_val)]
total_num_len=len(total_num)#5603个不同的记录
print(total_num_len)
# print(total_num)


a = np.array(mat, dtype=float)  # dtype指定为float


def graphMove(a):  # 构造转移矩阵
    b = transpose(a)  # b为a的转置矩阵
    c = zeros((a.shape), dtype=float)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            c[i][j] = a[i][j] / (b[j].sum())  # 完成初始化分配
    # print c,"\n===================================================="
    return c


def firstPr(c):  # pr值得初始化
    pr = zeros((c.shape[0], 1), dtype=float)  # 构造一个存放pr值得矩阵
    for i in range(c.shape[0]):
        pr[i] = float(1) / c.shape[0]
        # print pr,"\n==================================================="
    return pr


def pageRank(p, m, v):  # 计算pageRank值
    while ((v == p * dot(m, v) + (
        1 - p) * v).all() == False):  # 判断pr矩阵是否收敛,(v == p*dot(m,v) + (1-p)*v).all()判断前后的pr矩阵是否相等，若相等则停止循环
        # print v
        v = p * dot(m, v) + (1 - p) * v
        # print (v == p*dot(m,v) + (1-p)*v).all()
    return v


if __name__ == "__main__":
    M = graphMove(a)
    pr = firstPr(M)
    p = 0.85  # 引入浏览当前网页的概率为p,假设p=0.8
    ret=pageRank(p, M, pr)  # 计算pr值
    res=pd.DataFrame(ret)
    res.to_csv('numpy优化原理法最后PR值.csv')
    print('最终的PR值:\n', ret)
