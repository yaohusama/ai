import pandas as pd
# from collections import Counter
# res=pd.read_csv('出入关系矩阵.csv')
# for i in range(5603):
#     c=Counter(res[[i]])
#     print(c)
influence_data = pd.read_csv('influence_data.csv')
# c=Counter(influence_data['influencer_main_genre'])#计算各个出现量的数量
# tmp=influence_data[['influencer_id','follower_id']]
# print(type(tmp.iloc[0]))
# print(tmp[0]['influencer_id'])
min_val = 0x7fffffff
max_val = 0
print(len(influence_data['influencer_id']))  # 42770条数据
total_num = set()
unique_num = set()
max_val = 5603  # 42770条数据
mat = [[0] * max_val for i in range(max_val)]
for tmp in zip(influence_data['influencer_id'], influence_data['follower_id']):  # 还没筛选空值，不过从结果 来看没有空值
    total_num.add(tmp[0])
    total_num.add(tmp[1])
    tmp1 = min(tmp[0], tmp[1])
    min_val = min(min_val, tmp1)
    tmp1 = max(tmp[0], tmp[1])
    max_val = max(max_val, tmp1)
total_num = list(total_num)
for tmp in zip(influence_data['influencer_id'], influence_data['follower_id']):
    x = total_num.index(tmp[0])
    y = total_num.index(tmp[1])
    mat[x][y] = 1
    # print(x,y)
for i in mat:
    flag=False
    for j in i:
        if j==1:
            flag=True
            break
if flag:
    print("yes")