from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# pulls domain information from who is website
# need to pull data from the website


def domain_info(url):
    driver = webdriver.Firefox()
    driver.get("https://who.is/")
    assert "WHOIS Search, Domain Name, Website, and IP Tools - Who.is" in driver.title
    wait = WebDriverWait(driver, 10)
    elem = wait.until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/div[3]/div[1]/div/center/form/div/div/input')))
    elem.clear()
    elem.send_keys(url)
    elem.send_keys(Keys.RETURN)
    assert "WHOIS Search, Domain Name, Website, and IP Tools - Who.is" in driver.title
    print("The registry inforamation is as follows:")
    expiry_date = wait.until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/div[3]/div[2]/div[5]/div[1]/div[5]/div/div[1]/div[2]'))).text
    registered_date = wait.until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/div[3]/div[2]/div[5]/div[1]/div[5]/div/div[2]/div[2]'))).text
    updated_date = wait.until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/div[3]/div[2]/div[5]/div[1]/div[5]/div/div[3]/div[2]'))).text
    print("Will expire on: ", expiry_date)
    print("Registered on: ", registered_date)
    print("Updated on:", updated_date)
    driver.close()

# returns 0 for whitelisted websites
def black_listed(url):
    driver = webdriver.Firefox()
    driver.get("https://www.phishtank.com/")
    assert "PhishTank | Join the fight against phishing" in driver.title
    wait = WebDriverWait(driver, 10)
    elem = wait.until(EC.element_to_be_clickable((By.NAME, 'isaphishurl')))
    elem.clear()
    elem.send_keys(url)
    elem.send_keys(Keys.RETURN)
    assert "PhishTank | Join the fight against phishing" in driver.title
    text = wait.until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/div[2]/div[2]/div/div[1]/div/div/div[2]/form/p/b'))).text
    driver.close()
    if text[:19] == "Nothing known about":
        print("This site is not blacklisted")
    else:
        print("This site is blacklisted")

#runs the url against blacklist sources
def reputation_checker(url):
    driver = webdriver.Firefox()
    driver.get("https://www.urlvoid.com/")
    assert "Check if a Website is Malicious/Scam or Safe/Legit | URLVoid" in driver.title
    wait = WebDriverWait(driver, 10)
    elem = wait.until(EC.element_to_be_clickable((By.ID, 'hf-domain')))
    elem.clear()
    elem.send_keys(url)
    elem.send_keys(Keys.RETURN)
    assert "Check if a Website is Malicious/Scam or Safe/Legit | URLVoid" in driver.title
    text = wait.until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/div[3]/div[2]/div[2]/div/table/tbody/tr[3]/td[2]/span'))).text
    server_location = wait.until(EC.element_to_be_clickable(
        (By.XPATH, '/html/body/div[3]/div[2]/div[2]/div/table/tbody/tr[9]/td[2]'))).text
    driver.close()
    print("Blacklist score: ", text)
    print("Server Location: ", server_location)
# $prints page rank score on 10


def page_rank(url):
    driver = webdriver.Firefox()
    driver.get("https://dnschecker.org/pagerank.php")
    assert "Page Rank Checker - Check Your Website Pagerank" in driver.title
    wait = WebDriverWait(driver, 10)
    elem = wait.until(EC.element_to_be_clickable((By.ID, 'prc_host')))
    elem.clear()
    elem.send_keys(url)
    elem.send_keys(Keys.RETURN)
    assert "Page Rank Checker - Check Your Website Pagerank" in driver.title
    try :
        text = wait.until(EC.element_to_be_clickable(
            (By.XPATH, '/html/body/div[5]/div/div/div[1]/ul/li[2]/table/tbody/tr/td/span/span'))).text
        print("Page Rank score is :", text)
    except: 
        print("Domain not found for page rank")
    driver.close()


url1 = "platnosc-payu24.com/"
url2 = "google.com"

urls = [url1, url2]
for url in urls:
    print("Information from the Data Analyzer for :", url)
    domain_info(url)
    black_listed(url)
    reputation_checker(url)
    page_rank(url)
