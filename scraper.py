from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#pulls domain information from who is website 
#need to pull data from the website
def domain_info():
	driver = webdriver.Firefox()
	driver.get("https://who.is/")
	assert "WHOIS Search, Domain Name, Website, and IP Tools - Who.is" in driver.title
	wait = WebDriverWait(driver, 10)
	elem = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[3]/div[1]/div/center/form/div/div/input')))
	elem.clear()
	elem.send_keys('https://www.itflux.com/')
	elem.send_keys(Keys.RETURN)
	assert "WHOIS Search, Domain Name, Website, and IP Tools - Who.is" in driver.title
	driver.close()



#returns 0 for whitelisted websites
def white_listed():
	driver = webdriver.Firefox()
	driver.get("https://www.phishtank.com/")
	assert "PhishTank | Join the fight against phishing" in driver.title
	wait = WebDriverWait(driver, 10)
	elem = wait.until(EC.element_to_be_clickable((By.NAME, 'isaphishurl')))
	# elem = WebDriverWait(driver, 100).until(EC.element_to_be_clickable((By.NAME,"searchString")))
	elem.clear()
	elem.send_keys('www.google.com/')
	elem.send_keys(Keys.RETURN)
	assert "PhishTank | Join the fight against phishing" in driver.title

	text = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[2]/div[2]/div/div[1]/div/div/div[2]/form/p/b'))).text
	driver.close()
	print(text[:19])
	if  text[:19] == "Nothing known about":
	 	return 0
	return 1

#$prints page rank score on 10
def page_rank():
	driver = webdriver.Firefox()
	driver.get("https://dnschecker.org/pagerank.php")
	assert "Page Rank Checker - Check Your Website Pagerank" in driver.title
	wait = WebDriverWait(driver, 10)
	elem = wait.until(EC.element_to_be_clickable((By.ID, 'prc_host')))
	# elem = WebDriverWait(driver, 100).until(EC.element_to_be_clickable((By.NAME,"searchString")))
	elem.clear()
	elem.send_keys('www.google.com/')
	elem.send_keys(Keys.RETURN)
	assert "Page Rank Checker - Check Your Website Pagerank" in driver.title
	text = wait.until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[5]/div/div/div[1]/ul/li[2]/table/tbody/tr/td/span/span'))).text
	driver.close()
	print("Page Rank score is :", text)


domain_info()
flag2 = white_listed()
page_rank()