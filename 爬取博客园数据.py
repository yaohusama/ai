#coding=utf-8
import requests
from lxml import etree
'''
爬取内容：博客标题，摘要，发布时间，阅读量，评论数，推荐数
'''
url = 'https://www.cnblogs.com/pinard/default.html?page={}'
urls = [url.format(i) for i in range(1,15)]
#方法1：使用xpath解析html
for url in urls:
    print(url)
    response = requests.get(url)
    html = etree.HTML(response.text)
    blogs = html.xpath('//*[@id="mainContent"]/div/div')
    for blog in blogs:
        title = blog.xpath('./div[2]/a/span/text()')
        abstract = blog.xpath('./div[3]/div/text()')
        date = blog.xpath('./div[5]/text()')
        read = blog.xpath('./div[5]/span[1]/text()')
        pinlun = blog.xpath('./div[5]/span[2]/text()')
        tuijian = blog.xpath('./div[5]/span[3]/text()')
        if title:
            print(title,abstract,date,read,pinlun,tuijian)

#方法2：正则表达式解析html
#import re
#for url in urls:
    #response = requests.get(url)
    #html_str = response.text
    #html_str = re.sub('\s','',html_str)
    #blogs_pattern = r'<divclass="day">(.*?)<divclass="day"'          
    #blogs = re.findall(blogs_pattern, html_str)
    #for blog in blogs:
        #title = re.findall(r'postTitle2vertical-middle.*?<span>(.*?)</span>',blog)
        #abstract = re.findall(r'<divclass="c_b_p_desc">(.*?)<ahref',blog)
        #date = re.findall(r'posted@(.*?)刘建平',blog)
        #read = re.findall(r'阅读\((.*?)\)',blog)
        #pinlun = re.findall(r'评论\((.*?)\)',blog)
        #tuijian = re.findall(r'推荐\((.*?)\)',blog)   
        #print(title,abstract,date,read,pinlun,tuijian)
        
#方法3：使用bs4解析html
#from  bs4 import BeautifulSoup
#for url in urls:
    #response = requests.get(url)
    #soup = BeautifulSoup(response.text, 'lxml')
    #blogs = soup.select('.day')
    #for blog in blogs:
        #title = blog.select('.postTitle a span')[0].get_text()
        #abstract = blog.select('.postCon div')[0].get_text()
        #date = blog.select('.postDesc')[0]
        #read = blog.select('.post-view-count')[0].get_text()
        #pinlun = blog.select('.post-comment-count')[0].get_text()
        #tuijian = blog.select('.post-digg-count')[0].get_text()   
        #print(title,abstract,date,read,pinlun,tuijian)


