# -*- coding: utf-8 -*-

from pygraph.classes.digraph import digraph
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
mat=[[1] * max_val for i in range(max_val)]
for tmp in zip(influence_data['influencer_id'],influence_data['follower_id']):#还没筛选空值，不过从结果 来看没有空值
    total_num.add(tmp[0])
    total_num.add(tmp[1])
    tmp1=min(tmp[0],tmp[1])
    min_val=min(min_val,tmp1)
    tmp1=max(tmp[0],tmp[1])
    max_val=max(max_val,tmp1)
total_num=list(total_num)
# for tmp in zip(influence_data['influencer_id'], influence_data['follower_id']):
#     x=total_num.index(tmp[0])
#     y=total_num.index(tmp[1])
#     mat[x][y]=1


class PRIterator:
    __doc__ = '''计算一张图中的PR值'''

    def __init__(self, dg):
        self.damping_factor = 0.85  # 阻尼系数,即α
        self.max_iterations = 100  # 最大迭代次数
        self.min_delta = 0.00001  # 确定迭代是否结束的参数,即ϵ
        self.graph = dg

    def page_rank(self):
        #  先将图中没有出链的节点改为对所有节点都有出链
        for node in self.graph.nodes():
            if len(self.graph.neighbors(node)) == 0:
                for node2 in self.graph.nodes():
                    digraph.add_edge(self.graph, (node, node2))

        nodes = self.graph.nodes()
        graph_size = len(nodes)

        if graph_size == 0:
            return {}
        page_rank = dict.fromkeys(nodes, 1.0 / graph_size)  # 给每个节点赋予初始的PR值
        damping_value = (1.0 - self.damping_factor) / graph_size  # 公式中的(1−α)/N部分

        flag = False
        for i in range(self.max_iterations):
            change = 0
            for node in nodes:
                rank = 0
                for incident_page in self.graph.incidents(node):  # 遍历所有“入射”的页面
                    rank += self.damping_factor * (page_rank[incident_page] / len(self.graph.neighbors(incident_page)))
                rank += damping_value
                change += abs(page_rank[node] - rank)  # 绝对值
                page_rank[node] = rank

            print("This is NO.%s iteration" % (i + 1))
            print(page_rank)

            if change < self.min_delta:
                flag = True
                break
        if flag:
            print("finished in %s iterations!" % node)
        else:
            print("finished out of 100 iterations!")
        return page_rank


if __name__ == '__main__':
    dg = digraph()

    # dg.add_nodes(["A", "B", "C", "D", "E"])
    dg.add_nodes(total_num)
    # dg.add_edge(("A", "B"))
    # dg.add_edge(("A", "C"))
    # dg.add_edge(("A", "D"))
    # dg.add_edge(("B", "D"))
    # dg.add_edge(("C", "E"))
    # dg.add_edge(("D", "E"))
    # dg.add_edge(("B", "E"))
    # dg.add_edge(("E", "A"))
    for tmp in zip(influence_data['influencer_id'], influence_data['follower_id']):
    #     x = total_num.index(tmp[0])
    #     y = total_num.index(tmp[1])
        dg.add_edge((tmp[0],tmp[1]))

    pr = PRIterator(dg)
    page_ranks = pr.page_rank()

    print("The final page rank is\n", page_ranks)