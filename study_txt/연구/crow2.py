from selenium import webdriver
 
browser = webdriver.Chrome('C:/Users/chom_driver/chromedriver.exe')
browser.get('https://www.google.com')
browser.close()