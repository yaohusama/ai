import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import jieba
import re
'''G=nx.MultiDiGraph()#创建空的有向多图
pop_nodes=set()
evaluation_data=pd.read_csv('influence_data.csv',encoding='utf-8')
for  tmp in zip(evaluation_data['influencer_name'],evaluation_data['influencer_main_genre'],evaluation_data['follower_name'],evaluation_data['follower_main_genre']):
    if tmp[1]=='Electronic' and tmp[3]=='Electronic':
        pop_nodes.add(tmp[0])
        pop_nodes.add(tmp[2])
G.add_nodes_from(list(pop_nodes))
print(len(pop_nodes))
# for  tmp in zip(evaluation_data['influencer_name'],evaluation_data['influencer_main_genre'],evaluation_data['follower_name'],evaluation_data['follower_main_genre']):
#     if tmp[1]=='Pop/Rock' and tmp[3]=='Pop/Rock':
#         G.add_edge(tmp[0],tmp[2])

nx.draw_networkx(G)
plt.savefig('Electronic子网络不带边连接.png')
plt.show()'''



'''#子网络
G=nx.Graph()#创建空的简单图
# G=nx.DiGraph()#创建空的简单有向图
# G=nx.MultiGraph()#创建空的多图
G=nx.MultiDiGraph()#创建空的有向多图
evaluation_data=pd.read_excel('follower.xlsx')
# c=Counter(influence_data['influencer_main_genre'])#计算各个出现量的数量
# tmp=influence_data[['influencer_id','follower_id']]
# print(type(tmp.iloc[0]))
# print(tmp[0]['influencer_id'])
# min_val=0x7fffffff
# max_val=0
evaluation_nodes=evaluation_data['artist_name']
G.add_nodes_from(evaluation_nodes)
st=(evaluation_data['artist_name'][0])
# for i in range(1,6):
#     G.add_node(shortest_graph_list[i-1],position=(i,i))

for i in zip(evaluation_data['artist_name'],evaluation_data['weight']):
    if i[0] == st:
        continue
    if i[0]=='nan':
        continue
    print(i[0])
    tmp=i[1]
    tmp=round(tmp,2)
    G.add_edge(i[0],st,weight=tmp)
# pos  = nx.spring_layout(G, k=10)
# nx.draw(G, pos, with_labels=True)
nx.draw(G, alpha=0.5, with_labels=True,arrowsize=4)
# labels = nx.get_edge_attributes(G,'weight')
# nx.draw_networkx_edge_labels(G,pos)
# # nx.draw_networkx(G)
# plt.savefig('子网络.png')
# plt.show()
# nx.draw_networkx(G)
plt.savefig('子网络不带权值.png')
plt.show()'''
#一个路径图
G=nx.MultiDiGraph()#创建空的有向多图
influence_data=pd.read_csv('influence_data.csv',encoding='utf-8')
shortest_graph_list=['Blues', 'Country', 'Latin', 'Easy Listening', 'Avant-Garde']
G.add_nodes_from(shortest_graph_list)
# for i in range(1,6):
#     G.add_node(shortest_graph_list[i-1],position=(i,i))
G.add_edge('Blues', 'Country',weight=11)
G.add_edge('Country', 'Latin',weight=5)
G.add_edge('Latin', 'Easy Listening',weight=2)
G.add_edge('Easy Listening', 'Avant-Garde',weight=10)
# G.add_edge('Blues', 'Avant-Garde',weight=20.33)
pos= nx.spring_layout(G,k=10)
# nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G,'weight')
nx.draw_networkx_edge_labels(G,pos)
nx.draw_networkx(G,alpha=0.5,arrowsize=5)
plt.title('Blues_to_Avant-Grade')
plt.savefig('Blues_to_Avant-Grade.png')
plt.show()
# feature_1 = ['Boston', 'Boston', 'Chicago', 'ATX', 'NYC']
# feature_2 = ['LA', 'SFO', 'LA', 'ATX', 'NJ']
# score = ['1.00', '0.83', '0.34', '0.98', '0.89']
#
# df = pd.DataFrame({'f1': feature_1, 'f2': feature_2, 'score': score})
# print(df)
#
# G = nx.from_pandas_edgelist(df=df, source='f1', target='f2', edge_attr='score')
# pos = nx.spring_layout(G, k=10)  # For better example looking
# nx.draw(G, pos, with_labels=True)
# labels = {e: G.edges[e]['score'] for e in G.edges}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
#画出Eek-A-Mouse的跟随者和受到其影响的音乐家
'''beatles_followers_name=[]
beatles_followings_name=[]
for tmp in zip(influence_data['influencer_name'],influence_data['follower_name']):#还没筛选空值，不过从结果 来看没有空值
        if re.search('Eek-A-Mouse',tmp[0]) is not None:
                beatles_followers_name.append(tmp[1])
        if re.search('Eek-A-Mouse', tmp[1]) is not None:
                beatles_followings_name.append(tmp[0])
beatles_followers_name.append('Eek-A-Mouse')
G.add_nodes_from(beatles_followers_name)
for i in beatles_followers_name:
        G.add_edge('Eek-A-Mouse',i)
for i in beatles_followings_name:
        G.add_edge(i,'Eek-A-Mouse')
nx.draw_networkx(G)
plt.savefig('Eek-A-Mouse_followers_followings.png')
plt.show()'''
#只画Eek-A-Mouse影响的人
'''beatles_followers_name=[]
beatles_followings_name=[]
for tmp in zip(influence_data['influencer_name'],influence_data['follower_name']):#还没筛选空值，不过从结果 来看没有空值
        if re.search('Eek-A-Mouse',tmp[0]) is not None:
                beatles_followers_name.append(tmp[1])
beatles_followers_name.append('Eek-A-Mouse')
G.add_nodes_from(beatles_followers_name)
for i in beatles_followers_name:
        G.add_edge('Eek-A-Mouse',i)
nx.draw_networkx(G)
plt.savefig('Eek-A-Mouse_followers.png')
plt.show()'''
'''name_list=[]
total_num=set()
unique_num=set()
max_val=5603#42770条数据
mat=[[1] * max_val for i in range(max_val)]
for tmp in zip(influence_data['influencer_name'],influence_data['follower_name']):
    name_list.append(tmp[0])
    # name_list.append(tmp[1])
    # with open('name_list.txt','a+',encoding='utf-8') as f:
    #     f.write(tmp[0]+' ')
# for tmp in zip(influence_data['influencer_id'],influence_data['follower_id']):#还没筛选空值，不过从结果 来看没有空值
#     total_num.add(tmp[0])
#     total_num.add(tmp[1])
#     tmp1=min(tmp[0],tmp[1])
#     min_val=min(min_val,tmp1)
#     tmp1=max(tmp[0],tmp[1])
#     max_val=max(max_val,tmp1)
# print(max_val)
# total_num=list(total_num)
# G.add_nodes_from(total_num)
# for tmp in zip(influence_data['influencer_id'],influence_data['follower_id']):#还没筛选空值，不过从结果 来看没有空值
#     G.add_edge(tmp[0],tmp[1])
#
# nx.draw_networkx(G)
# plt.show()#绘制全连接的图
import matplotlib.pyplot as plt

from wordcloud import WordCloud


# # 1.读入txt文本数据
# text = open(r'name_list.txt', "r",encoding='utf-8').read()
# #print(text)
# # 2.结巴中文分词，生成字符串，默认精确模式，如果不通过分词，无法直接生成正确的中文词云
# cut_text = jieba.cut(text)
# # print(type(cut_text))
# # 必须给个符号分隔开分词结果来形成字符串,否则不能绘制词云
result = " ".join(name_list)
#print(result)
# 3.生成词云图，这里需要注意的是WordCloud默认不支持中文，所以这里需已下载好的中文字库
# 无自定义背景图：需要指定生成词云图的像素大小，默认背景颜色为黑色,统一文字颜色：mode='RGBA'和colormap='pink'
wc = WordCloud(
        # 设置字体，不指定就会出现乱码
        # 设置背景色
        background_color='white',
        # 设置背景宽
        width=500,
        # 设置背景高
        height=350,
        # 最大字体
        max_font_size=50,
        # 最小字体
        min_font_size=10,
        mode='RGBA'
        #colormap='pink'
        )
# 产生词云
wc.generate(result)
# 保存图片
wc.to_file(r"wordcloud.png") # 按照设置的像素宽高度保存绘制好的词云图，比下面程序显示更清晰
# 4.显示图片
# 指定所绘图名称
plt.figure("jay")
# 以图片的形式显示词云
plt.imshow(wc)
# 关闭图像坐标系
plt.axis("off")
plt.show()
'''
