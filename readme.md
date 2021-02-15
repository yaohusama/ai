+ 本仓库部分文件使用了LFS，大文件传输，如果要克隆需要先安装LFS，教程我在https://blog.csdn.net/qq_42865713/article/details/113812472写过。
+ water_and_fertilizer_CV是使用opencv，滑窗和PCA降维,adaboost的基于图像处理的智慧水肥喷洒系统，大创项目。本人负责内容是将其他人整理的资料结合自己学习的部分，写成一个接口，读取images文件夹中的图片，预处理中值滤波，放缩图片以减少数据量，采取直方图自适应均衡化，改变亮度。提取图片黄色和绿色部分，使用形态学操作将其中黑色背景中的白点，前景中的背景，去除，只是去除了其中略小的部分。主要负责调参加代码部分。使用adaboost自写的代码（参考博客）对已打好的标签和之前提取的特征训练，调整参数，主要是多少个弱学习器，弱学习器的阈值调整的步长，使得训练集错误率在0，其实用官方接口也是可以达到，但是为了在论文中更详细地说明参数的调整过程，就用了自定义的代码，且将总体抽出了0.3作为测试集，根据滑窗大小从1~19来计算错误率，从而确定合适的窗口大小。颜色矩是最明显的特征，但是不清楚是否滑窗会影响灰度共生矩阵（纹理特征），不变矩（形状特征），这两类，所以滑窗只应用在了对结果从主观上影响最明显的颜色矩，文件夹有写成的论文介绍，但2月15日之前还未拿出硬件，也没有在树莓派中跑过，个人觉得之后自己可能会参与其中，但是由于经费有限，两个树莓派，并没有1个在我这儿，对硬件涉猎不深。
+ 比赛中的美赛是2021年2月，代码跑通，整篇根据数据三个人一起写的，使用了PR算法，灰色综合评价，聚类，降维，还有一些较为直观的分析方法。本人负责还是主要代码部分还有画图，提取数据特征。PR算法假设了如果一个节点没有指向任何节点，就假设都连上，实际上不符合题意，导致生成的PR值，和事实有些许不符，然后根据影响者的词频来生成词云，可视化的结果很好。使用了灰色综合评价法，评价对某人影响最大的影响者，这一块我只提取了数据，不过这个耗时的确挺长的，后来的调包生成结果没弄，当时队长写的论文模板有些许问题，便放置在那儿了。PR使用了map,reduce来加快程序，如若不用，似乎跑很久也不一定跑出来。使用networkx作图，并且经过此次美赛，意识到excel作图又方便又美观。剩下的就是大量的可视化和论文部分的撰写，队友对此作了很大贡献。
+ 比赛中的数独比赛，是丝路杯，几乎没用到，因为考试时网络崩了，结果新的考试时间说了短信通知又没通知，最后只有补考。使用C++编写DFS解数独（参考网上代码），对文件中的其他类型数独，将网上代码进行了更改。
+ 小程序中的知乎小程序是github上克隆的样例，本人自做的小程序是浴室小程序，队友给了各个界面设计图，我自己伪造了本地数据，编写了前端部分，浴室小程序在学校里可能只能到查看浴室情况那一块儿，因为学校不允许对浴室门进行破坏性改装，预约功能可能比较鸡肋，所以服务端暂未开发。总体是获取用户授权后，再转到个人信息界面，点下确认后，将获得的数据缓存到本地（理应同时提交给服务器，但暂且没有），再转到查询页面，当前楼层人满和人没满分两个逻辑，人满的时候需要预约，没满，直接选浴室即可。
![img_1.png](img_1.png)
  