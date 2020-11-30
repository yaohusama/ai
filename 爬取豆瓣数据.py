#coding=utf-8
import requests
'''
id,title,actors,score,types,rank,release_date,cover_url
'''
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36"}

start_url = 'https://movie.douban.com/j/chart/top_list?type=24&interval_id=100%3A90&action=&start={}&limit=20'
page = 0

while True:
    url = start_url.format(page)
    print(url)
    res = requests.get(url,headers=headers)
    print(res.status_code)
    movies = res.json()
    if not movies:
        print(page)
        break
    for movie in movies:
        print(movie['id'])
        print(movie['title'])
        print(movie['actors'])
        print(movie['score'])
        print(movie['types'])
        print(movie['rank'])
        print(movie['release_date'])
        print(movie['cover_url'])
    page += 20
    