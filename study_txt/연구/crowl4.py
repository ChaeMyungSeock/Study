  
from bs4 import BeautifulSoup as bs
from urllib.request import urlopen
from urllib.parse import quote_plus
 
# BeautifulSoup 라이브러리는 HTML 및 XML 파일에서 데이터를 가져 오는 Python 라이브러리
baseUrl = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query='
# https://search.naver.com/search.naver?where=image&sm=tab_jum&query= 라는 주소에 검색어만 바뀌는 것을 확인

plusUrl = input('검색어 입력: ') 
crawl_num = int(input('크롤링할 갯수 입력(최대 50개): '))
# 네이버 이미지는 한페이지에 50개의 이미지만 생성되므로 보통 이미지가 다운로드되면 50개까지 가능함

# url = baseurl+ 검색어
url = baseUrl + quote_plus(plusUrl) # 한글 검색 자동 변환
# html을 열어서 이미지의 경로를 받아옵니다
html = urlopen(url)
soup = bs(html, "html.parser")
img = soup.find_all(class_='_img')
 
n = 100
for i in img:
    print(n)
    imgUrl = i['data-source']
    with urlopen(imgUrl) as f:
        with open('F:/Study/태연/' + str(n)+'.jpg','wb') as h: # w - write b - binary
            img = f.read()
            h.write(img)
    n += 1
    if n > crawl_num+100:
        break
print('Image Crawling is done.')