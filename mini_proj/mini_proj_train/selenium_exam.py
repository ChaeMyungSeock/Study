from selenium import webdriver
path = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"
driver = webdriver.Chrome(path)

'''
보통 한 요소를 리턴하는 find_element_*()
특정 태그 id 로 검색하는 find_element_by_id(), 
특정 태그 name 속성으로 검색하는 find_element_by_name(),
CSS 클래스명으로 검색하는 find_element_by_class_name(),
CSS selector를 사용해 검색하는 find_element_by_css_selector() 등이 있는데, 
예상되는 결과가 복수이면 find_element_* 대신 find_elements_* 를 사용한다.

'''