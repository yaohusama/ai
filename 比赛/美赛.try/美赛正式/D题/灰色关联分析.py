import pandas as pd
import numpy as np
'''pr_data=pd.read_csv('mapreduce法最后PR值.csv',encoding='utf-8')#没有去除重复的记录
pr_func={}
for i in zip(pr_data['id'],pr_data['PR']):
    pr_func[i[0]]=i[1]
# pr_col=pr_data['PR']
# print(pr_col)
influence_data=pd.read_csv('influence_data.csv',encoding='utf-8')
count=0
total_num=set()
# unique_artist=set()
genre_dict= {}
year_dict= {}
bool_dict= {}
influence_people_number={}
influencer_dict={}
unique_num=set()
st_min=10000
st_max=1000
st_set=set()
influence_dict={}
# max_val=5603#42770条数据
# mat=[[0] * max_val for i in range(max_val)]#有可能艺术家会出现多次，所以要去重，甚至会出现同一个艺术家却不同流派，以防万一以下
for influencer in zip(influence_data['influencer_id'],influence_data['influencer_main_genre'],influence_data['influencer_active_start'],influence_data['follower_id'],influence_data['follower_main_genre'],influence_data['follower_active_start']):
    # unique_artist.add((influencer[0],influencer[1],influencer[2]))
    # unique_artist.add((influencer[3], influencer[4], influencer[5]))
    total_num.add(influencer[0])
    total_num.add(influencer[3])
    # st_min=min(st_min,influencer[2])
    # st_min = min(st_min, influencer[5])
    # st_max = max(st_max, influencer[2])
    # st_max = max(st_max, influencer[5])
    st_set.add(influencer[2])
    if influencer[2] not in influence_dict.keys():
        influence_dict[influencer[2]]=[]
        influence_dict[influencer[2]].append(influencer[0])
        # print(influence_dict[influencer[2]])
    else :
        influence_dict[influencer[2]].append(influencer[0])
        # print(influence_dict[influencer[2]])
# print(influence_dict)
    # st_set.add(influencer[5])
   # print(influencer[0],influencer[3])
#for influencer in zip(influence_data['influencer_name'],influence_data['influencer_main_genre'],influence_data['follower_name'],influence_data['follower_main_genre']):
    # count=count+1
    # if count==1:
    #     continue
    # print(influencer)
# for influence in total_num:
#     bool_dict[influence]=0
# rk={}
# for time in st_set:
#     for tmp in influence_dict[time]:
#         for tmp1 in tmp:
#             rk[tmp1]=tmp.count(tmp1)
#     sort(rk,key=rk.values(),ascending=True)
#
for influencer in zip(influence_data['influencer_id'],influence_data['influencer_main_genre'],influence_data['influencer_active_start'],influence_data['follower_id'],influence_data['follower_main_genre'],influence_data['follower_active_start']):
    # print(influencer[0])
    # print(influencer[3])
    # if bool_dict[influencer[0]]:
    #     continue
    genre_dict[influencer[0]]=influencer[1]
    influencer_dict[influencer[0]]=influencer[2]
    year_dict[influencer[0]]=influencer[2]
    # bool_dict[influencer[0]]=1
    # if bool_dict[influencer[3]]:
    #     continue
    genre_dict[influencer[3]] = influencer[4]
    year_dict[influencer[3]] = influencer[5]
    # bool_dict[influencer[3]] = 1
# print(total_num)
# print(genre_dict[3637248])
x2=[]
x1=[]
x3=[]
x4=[]
x5=[]
print('-------------------------------------------------------------')
for tmp in total_num:
    # print(tmp)
    influencer_genre=genre_dict[tmp]#'influencer_main_genre'
    # print(influencer_genre)
    influencer_year=year_dict[tmp]
    influence_genre_count=list(genre_dict.values()).count(influencer_genre)#len(influence_data[influence_data['influencer_main_genre']==influencer_genre])
    # print(influence_genre_count)
    x3_tmp= {}
    for i in influencer_dict.keys():
        if genre_dict[i]==influencer_genre:
            x3_tmp[i]=year_dict[i]
    # year_genre_dict=year_dict[genre_dict[tmp]==influencer_genre]
    influence_year_genre_count=list(x3_tmp.values()).count(influencer_year)#len(influence_data[influence_data['influencer_main_genre']==influencer_genre])
    influence_year_count=list(year_dict.values()).count(influencer_year)
    x2.append(influence_genre_count)
    x1.append(pr_func[tmp])
    x3.append(influence_year_genre_count)
    x5.append(influence_year_count)
    rk={}
    for tmp1 in influence_dict[influencer_year]:
        # print(tmp1)
        rk[tmp1]=influence_dict[influencer_year].count(tmp1)
        # for tmp2 in tmp1:
        #     rk[tmp2] = tmp1.count(tmp2)
    # rk=sorted(rk.items(),key=lambda x:x[1], reverse=True)
    print(rk)
    if tmp not in rk.keys():
        x4.append(0)
    else :
        x4.append(rk[tmp])
    # influence_people_number[tmp]=influence_data['influencer_id'].count(tmp)
# for tmp in st_set:
#     # print(tmp)
#     influencer_genre=genre_dict[tmp]#'influencer_main_genre'
#     # print(influencer_genre)
#     influencer_year=year_dict[tmp]
#     rk={}
#     for tmp1 in influence_dict[influencer_year]:
#         # print(tmp1)
#         rk[tmp1]=influence_dict[influencer_year].count(tmp1)
#         # for tmp2 in tmp1:
#         #     rk[tmp2] = tmp1.count(tmp2)
#     rk=sorted(rk.items(),key=lambda x:x[1], reverse=True)
#     x4.append(rk[tmp])
#     # influence_people_number[tmp]=influence_data['influencer_id'].count(tmp)
res={'x1':x1,'x2':x2,'x3':x3,'x4':x4,'x5':x5}
res=pd.DataFrame(res,index=total_num)
res.to_csv('灰色综合评价法的五个指标.csv',encoding='utf-8')'''
# print(x1,x2)
import pandas as pd
x=pd.read_csv('灰色综合评价法的五个指标.csv',encoding='utf-8')
ck=[x['x1'].max(),x['x2'].max(),x['x3'].max(),x['x4'].max(),x['x5'].max()]
# lis=x['x1']
x=x.iloc[:,1:].T

# 1、数据均值化处理
x_mean=x.mean(axis=1)
for i in range(x.index.size):
    x.iloc[i,:] = x.iloc[i,:]/x_mean[i]

# 2、提取参考队列和比较队列

cp=x.iloc[:,:]
print(cp)
# 比较队列与参考队列相减
t=pd.DataFrame()
for j in range(cp.index.size):
    temp=pd.Series(cp.iloc[j,:]-ck)
    t=t.append(temp,ignore_index=True)

#求最大差和最小差
mmax=t.abs().max().max()
mmin=t.abs().min().min()
rho=0.5
#3、求关联系数
ksi=((mmin+rho*mmax)/(abs(t)+rho*mmax))


#4、求关联度
r=ksi.sum(axis=1)/ksi.columns.size

#5、关联度排序，得到结果r3>r2>r1
result=r.sort_values(ascending=False)
print(result)
