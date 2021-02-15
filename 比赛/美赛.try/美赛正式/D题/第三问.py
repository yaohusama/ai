from pygraph.classes.digraph import digraph
# -*- coding: utf-8 -*-
from numpy import *
import pandas as pd
import networkx as nx
from pygraph.classes.digraph import digraph
from PRMapReduce import PRMapReduce
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from collections import Counter
#
influence_data=pd.read_csv('influence_data.csv',encoding='utf-8')
# c=Counter(influence_data['influencer_main_genre'])#计算各个出现量的数量
# tmp=influence_data[['influencer_id','follower_id']]
# print(type(tmp.iloc[0]))
# print(tmp[0]['influencer_id'])
#
print(len(influence_data['influencer_id']))#42770条数据
pai_tuple=set()
total_num=set()
unique_num=set()
max_val=20#42770条数据
mat=[[0] * max_val for i in range(max_val)]
count_mat=[[0] * max_val for i in range(max_val)]
pai_dict={}
pai_tuple=tuple(np.unique(influence_data['influencer_main_genre']))
pai_tuple=np.unique(pai_tuple+tuple(np.unique(influence_data['follower_main_genre'])))#加号可以连接元组
pai_tuple=list(pai_tuple)
for i,value in enumerate(pai_tuple):
    pai_dict[value]=i
print(pai_dict)

# print(pai_tuple)
# print(len(pai_tuple))
G=nx.MultiDiGraph()#创建空的有向多图
for tmp in zip(influence_data['influencer_main_genre'],influence_data['follower_main_genre'],influence_data['follower_active_start'],influence_data['influencer_active_start']):#还没筛选空值，不过从结果 来看没有空值

    mat[pai_dict[tmp[0]]][pai_dict[tmp[1]]]=mat[pai_dict[tmp[0]]][pai_dict[tmp[1]]]+(tmp[2]-tmp[3])
    count_mat[pai_dict[tmp[0]]][pai_dict[tmp[1]]]=count_mat[pai_dict[tmp[0]]][pai_dict[tmp[1]]]+1
for i in range(20):
    for  j in range(20):
        if mat[i][j] < 0:
            mat[i][j]=0
        if mat[i][j]!=0:
            mat[i][j]=mat[i][j]/count_mat[i][j]
G.add_nodes_from(pai_tuple)
print(pai_tuple)
pai_weight=[]
for i in range(20):
    for j in range(20):
        if(mat[i][j]!=0):
            if i==j:
                mat[i][i]=0#同一个传过去是0
            pai_weight.append((pai_tuple[i],pai_tuple[j],mat[i][j]))
            G.add_edge(pai_tuple[i],pai_tuple[j],mat[i][j])

nx.draw_networkx(G,alpha=0.5,arrowsize=4)
plt.rcParams['font.sans-serif']=['Simhei']
plt.title('流派演化')
plt.savefig('流派演化.png')
plt.show()
'''INF_val = 9999


class Floyd_Path():
    def __init__(self, node, node_map, path_map):
        self.node = node
        self.node_map = node_map
        self.node_length = len(node_map)
        self.path_map = path_map
        self._init_Floyd()

    def __call__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        return self._format_path()

    def _init_Floyd(self):
        for k in range(self.node_length):
            for i in range(self.node_length):
                for j in range(self.node_length):
                    tmp = self.node_map[i][k] + self.node_map[k][j]
                    if self.node_map[i][j] > tmp:
                        self.node_map[i][j] = tmp
                        self.path_map[i][j] = self.path_map[i][k]

        print('_init_Floyd is end')

    def _format_path(self):
        node_list = []
        temp_node = self.from_node
        obj_node = self.to_node
        # print(("the shortest path is: %d") % (self.node_map[temp_node][obj_node]))
        # res=self.node_map[temp_node][obj_node]
        # node_list.append(res)
        node_list.append(self.node[temp_node])
        while True:
            node_list.append(self.node[self.path_map[temp_node][obj_node]])
            temp_node = self.path_map[temp_node][obj_node]
            if temp_node == obj_node:
                break

        return node_list


def set_node_map(node_map, node, node_list, path_map):
    for i in range(len(node)):
        ## 对角线为0
        node_map[i][i] = 0
    for x, y, val in node_list:
        node_map[node.index(x)][node.index(y)] = val
        path_map[node.index(x)][node.index(y)] = node.index(y)
        path_map[node.index(y)][node.index(x)] = node.index(x)


# if __name__ == "__main__":
# node = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
node=pai_tuple
# node_list = [('A', 'F', 9), ('A', 'B', 10), ('A', 'G', 15), ('B', 'F', 2),
#              ('G', 'F', 3), ('G', 'E', 12), ('G', 'C', 10), ('C', 'E', 1),
#              ('E', 'D', 7)]

node_list=pai_weight
## node_map[i][j] 存储i到j的最短距离
node_map = [[INF_val for val in range(len(node))] for val in range(len(node))]
# for i in range(20):
#     node_map[i][i]=0
# node_map=[0 for i in range(20) for j in range(20) if i==j else INF_val]
print(node_map)
## path_map[i][j]=j 表示i到j的最短路径是经过顶点j
path_map = [[0 for val in range(len(node))] for val in range(len(node))]

## set node_map
set_node_map(node_map, node, node_list, path_map)

## select one node to obj node, e.g. A --> D(node[0] --> node[3])
from_node = node.index('Avant-Garde')
to_node = node.index('Blues')
Floydpath = Floyd_Path(node, node_map, path_map)
# a=Floydpath(from_node,to_node)
import sys
shortest_path=[[sys.maxsize] * max_val for i in range(max_val)]
path=[[[]] * max_val for i in range(max_val)]

for  i in range(20):
    for j in range(20):
        path[i][j] = Floydpath(node.index(pai_tuple[i]), node.index(pai_tuple[j]))
shortest_path=Floydpath.node_map
shortest_path=pd.DataFrame(shortest_path,columns=pai_tuple,index=pai_tuple)
print(shortest_path)
shortest_path.to_csv('shortest_length_between_genres.csv',encoding='utf-8')
path=pd.DataFrame(path,columns=pai_tuple,index=pai_tuple)
print(path)
path.to_csv('shortest_path_between_genres.csv',encoding='utf-8')'''
# print(path)
# nx.draw_networkx(G)
# plt.rcParams['font.sans-serif']=['Simhei']
# plt.title('流派演化')
# plt.savefig('流派演化.png')
# plt.show()
# import sys

# def floyd(l, n):
#     '''
#     l: l[i][j] = distace of i and j if <i, j> in E else sys.maxsize
#     k: sum of point
#     '''
#     d = l[:]
#
#     route = [([''] * n) for i in range(n)]
#     for i in range(n):
#         for j in range(n):
#             if d[i][j]:
#                 route[i][j] = str(i + 1) + " " + str(j + 1)
#
#     for k in range(n):
#         for i in range(n):
#             for j in range(n):
#                 if d[i][j] > d[i][k] + d[k][j]:
#                     d[i][j] = d[i][k] + d[k][j]
#                     route[i][j] = route[i][k] + " " + route[k][j][2:]
#     return d, route
#
# # total_num=list(total_num)
#
# if __name__ == "__main__":
#     n = 3
#     l = [[0, 2, 9], [8, 0, 6], [1, sys.maxsize, 0]]
#     d, route = floyd(l, n)
#
#     for i in d:
#         for j in i:
#             print
#             j,
#         print
#         ""
#
#     for i in route:
#         for j in i:
#             print
#             "[" + j + "],",
#         print
#         ""

