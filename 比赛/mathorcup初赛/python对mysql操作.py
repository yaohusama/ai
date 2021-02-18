#coding=utf-8
import pymysql
import fileinput
# ##连接mysql数据库
# mysqldb = pymysql.connect(host='localhost',
#                           user='root',
#                           password='',
#                           database = 'xtt',
#                           port=3306)
# # #获取游标
# cursor = mysqldb.cursor()
import pymysql
import fileinput
import pandas as pd
import numpy as np
##连接mysql数据库
res=pd.read_csv('duanqi.csv',encoding='gbk')
res=pd.DataFrame(res)
res1=res['小区编号']
res1=np.unique(res1)
create_table_sql = '''
use xtt;
create table IF NOT EXISTS changqi (
cq_id long NOT NULL AUTO_INCREMENT,
cq_date datetime NOT NULL,
bh long NOT NULL,
PRIMARY KEY(cq_id),
FULLTEXT(bh)
)ENGINE=INNODB;
'''
from mysql_operater import PythonMysql
mysql_object = PythonMysql('localhost', 'root', '','xtt', 3306)
if mysql_object:
    cursor1 = mysql_object
    for i in res1:
        sql = '''
        select tr_date,tr_time,upgb,downgb from train
        where bh = {}
        '''.format(i)
        res=cursor1.select_mysql(sql)
        # with open('{}.csv'.format(i),'w') as f:
        #     f.write(res)
        name = []
        results_list = list(res)
        name=['tr_date','tr_time','upgb','downgb']
        file_test = pd.DataFrame(columns=name, data=results_list)
        file_test.to_csv('xunlianji/{}.csv'.format(i), encoding='utf-8', index=False)