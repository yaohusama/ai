from pygraph.classes.digraph import digraph
# -*- coding: utf-8 -*-
from numpy import *
import pandas as pd
from pygraph.classes.digraph import digraph
# -*- coding: utf-8 -*-
from numpy import *
import pandas as pd
from PRMapReduce import PRMapReduce
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

    pr = PRMapReduce(dg)
    page_ranks = pr.page_rank()

    print("The final page rank is")
    keys=[]
    values=[]
    for key, value in page_ranks.items():
        keys.append(key)
        values.append(value[0])
        print(str(key) + " : ", value[0])
    # res = pd.DataFrame(keys,values)
    res={"id":keys,"PR":values}
    print(values)
    res=pd.DataFrame(res)
    res.to_csv('mapreduce法最后PR值.csv')