import pandas as pd
import matplotlib.pyplot as plt
'''influence_data=pd.read_csv('influence_data.csv',encoding='utf-8')
pr_list={}

main_genre=set()
for tmp in zip(influence_data['influencer_main_genre'],influence_data['follower_main_genre']):
    main_genre.add(tmp[0])
    main_genre.add(tmp[1])
x_len=[]
def pr_part(i):
    pop_id = set()
    for tmp in zip(influence_data['influencer_id'], influence_data['influencer_main_genre'],
                   influence_data['follower_id'], influence_data['follower_main_genre']):
        if tmp[1] == i:
            pop_id.add(tmp[0])
        if tmp[3] == i:
            pop_id.add(tmp[2])
    pr_data = pd.read_csv('mapreduce法最后PR值.csv', encoding='utf-8')
    pr_list[i] = []
    for tmp in zip(pr_data['id'], pr_data['PR']):
        if tmp[0] in pop_id:
            pr_list[i].append(tmp[1])
    if i=='Pop/Rock':
        x_len.append(len(pr_list[i]))
    pr_list[i]=sorted(pr_list[i])
for tmp in main_genre:
    pr_part(tmp)

import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color
import random
import matplotlib.pyplot as plt

# sub_axix = filter(lambda x:x%200 == 0, x_axix)
x_axix=range(x_len[0])
plt.rcParams['font.sans-serif']=['SimHei']
plt.title('PR分布')
for tmp in main_genre:
    plt.plot(range(len(pr_list[tmp])), pr_list[tmp], color=randomcolor(), label=tmp)

# plt.plot(x_axix, test_acys, color='red', label='testing accuracy')
#
# plt.plot(x_axix, train_pn_dis, color='skyblue', label='PN distance')
#
# plt.plot(x_axix, thresholds, color='blue', label='threshold')

plt.legend() # 显示图例

plt.xlabel('个')

plt.ylabel('PR')
plt.savefig('PR分布')
plt.show()'''
# pr_pop_len = len(pr_list)
# pr_list[i] = sorted(pr_list)  # 默认从小到大
# plt.plot(range(pr_pop_len), pr_list)
# plt.show()
# main_genre=set()
influence_data=pd.read_csv('influence_data.csv',encoding='utf-8')
main_genre=set()
for tmp in zip(influence_data['influencer_main_genre'],influence_data['follower_main_genre']):
    main_genre.add(tmp[0])
    main_genre.add(tmp[1])
genre_count={}
genre_min=[]
genre_num={}
count_num=0
for tmp in main_genre:
    tmp1=list(influence_data['influencer_main_genre']).count(tmp)
    genre_count[tmp]=tmp1
    genre_num[tmp]=count_num
    # print(tmp,count_num)
    if tmp1<=100:
        genre_min.append(tmp1)
    count_num=count_num+1
print(genre_count)
genre_count=pd.DataFrame(list(genre_count.values()),index=list(genre_count.keys()))
genre_count.to_csv('各派别的人数.csv',encoding='utf-8')
# print(genre_count['Religious'])
# # genre_count=sorted(genre_count.items(),key=lambda x:x[1])
# main_genre=list(main_genre)
'''import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.figure(figsize=(12,12))#将画布设定为正方形，则绘制的饼图是正圆
# label=['第一','第二','第三']#定义饼图的标签，标签是列表
# for i in range
# label=genre_count[0]
# for i in range
# label=main_genre
explode=[0.02]*len(main_genre)#设定各项距离圆心n个半径
elements = main_genre
for tmp in main_genre:
    print(genre_num[tmp])
    explode[genre_num[tmp]]=0.04
#plt.pie(values[-1,3:6],explode=explode,labels=label,autopct='%1.1f%%')#绘制饼图
# values=[4,7,9]
values=list(genre_count.values())
wedges=plt.pie(values,explode=explode,pctdistance=0.9)#绘制饼图,labels=label,autopct='%1.1f%%'

# patches,l_text,p_text=
# for t in p_text:
#     t.set_size(10)
#
# for t in l_text:
#     t.set_size(10)
# plt.legend(wedges,
#            elements,
#            fontsize=12,
#            title="配料图",
#            loc="center left")
plt.title('各领域艺术家分布')#绘制标题
plt.savefig('各领域艺术家分布.png')#保存图片
plt.show()'''