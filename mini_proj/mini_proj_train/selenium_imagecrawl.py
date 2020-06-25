from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import os
import urllib.request
 
#찾고자 하는 검색어를 url로 만들어 준다.
searchterm = "IU"
url = "https://www.google.com/search?q="+searchterm+"&source=lnms&tbm=isch"
# chrom webdriver 사용하여 브라우저를 가져온다.
browser = webdriver.Chrome('C:/Users/bitcamp/Desktop/program/chromedriver_win32/chromedriver.exe')
browser.get(url)

# User-Agent를 통해 봇이 아닌 유저정보라는 것을 위해 사용
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"}
# 이미지 카운트 (이미지 저장할 때 사용하기 위해서)
counter = 0
succounter = 0
 
print(os.path)
# 소스코드가 있는 경로에 '검색어' 폴더가 없으면 만들어준다.(이미지 저장 폴더를 위해서) 
if not os.path.exists(searchterm):
    os.mkdir(searchterm)
 
for _ in range(500):
    # 가로 = 0, 세로 = 10000 픽셀 스크롤한다.
    browser.execute_script("window.scrollBy(1000,1000)")
# div태그에서 class name이 rg_meta인 것을 찾아온다
for x in browser.find_elements_by_xpath('//div[contains(@class,"rg_meta")]'):
    counter = counter + 1
    print ("Total Count:", counter)
    print ("Succsessful Count:", succounter)
    print ("URL:",json.loads(x.get_attribute('innerHTML'))["ou"])
 
    # 이미지 url
    img = json.loads(x.get_attribute('innerHTML'))["ou"]
    # 이미지 확장자
    imgtype = json.loads(x.get_attribute('innerHTML'))["ity"]
    
    # 구글 이미지를 읽고 저장한다.
    try:
        req = urllib.request(img, headers={'User-Agent': header})
        raw_img = urllib.request.urlopen(req).read()
        File = open(os.path.join(searchterm , searchterm + "_" + str(counter) + "." + imgtype), "wb")
        File.write(raw_img)
        File.close()
        succounter = succounter + 1
    except:
            print ("can't get img")
            
print (succounter, "succesfully downloaded")
browser.close()
 