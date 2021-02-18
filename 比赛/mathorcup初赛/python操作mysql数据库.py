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


select_sql = '''
select t1.student_name,t2.age,t2.phone from ai.student t1
join 
ai.student1 t2 on t1.num=t2.num where t1.english>70 order by age,phone
'''
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
create_table_sql='''
use xtt;
CREATE TABLE IF NOT EXISTS tasks (
  task_id INT(11) NOT NULL AUTO_INCREMENT,
  subject VARCHAR(45) DEFAULT NULL,
  start_date DATE DEFAULT NULL,
  end_date DATE DEFAULT NULL,
  description VARCHAR(200) DEFAULT NULL,
  PRIMARY KEY (task_id)
) ENGINE=InnoDB;
'''

##通过游标执行查询语句
#cursor.execute(select_sql)
##通过游标获取所有查询结果
#data = cursor.fetchall()
#print(data)
#cursor.close()
from mysql_operater import PythonMysql
mysql_object = PythonMysql('localhost', 'root', '','xtt', 3306)
#mysql_object.execute("use xtt;insert into changqi values(NULL,0,1);")
# print("hello")
if mysql_object:
    cursor1 = mysql_object
    # cursor.create_table(create_table_sql)
    # for line in fileinput.input("train.txt"):
    #     lines=line.split(',')
    #     #lines[0]=lines[0].replace('/','-')
    #     insert_sql =r"insert into train values(NULL,'{0}',{1},{2},{3});".format(lines[0]+' '+lines[1],lines[2],lines[3],lines[4])
    #     print(insert_sql)
    #     cursor.insert_mysql(insert_sql)
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
        ##建立DataFrame对象
        file_test = pd.DataFrame(columns=name, data=results_list)
        ##数据写入,不要索引
        file_test.to_csv('xunlianji/{}.csv'.format(i), encoding='utf-8', index=False)