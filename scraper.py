from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
from https import check_all
import datetime
from neural_network_helper import extract_attributes_from_file
from https import format_url
from neural_network_helper import get_samples_with_urlvoid_fail

MAX_TRIES = 1
SCRAPER = ""#chrome"

# pulls domain information from who is website
# need to pull data from the website
#driver = webdriver.Chrome()

def domain_info(url, count = 0):
    opendriver = False
    while count < MAX_TRIES:
        try:
            if SCRAPER == "chrome":
                driver = webdriver.Chrome()
            else:
                driver = webdriver.Firefox()
            #driver.minimize_window()
            opendriver = True
            driver.get("https://who.is/")
            assert "WHOIS Search, Domain Name, Website, and IP Tools - Who.is" in driver.title
            wait = WebDriverWait(driver, 10)
            elem = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[3]/div[1]/div/center/form/div/div/input')))
            elem.clear()
            elem.send_keys(url)
            elem.send_keys(Keys.RETURN)
            #assert "WHOIS Search, Domain Name, Website, and IP Tools - Who.is" in driver.title
            print("The registry inforamation is as follows:")
            expiry_date = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[3]/div[2]/div[5]/div[1]/div[5]/div/div[1]/div[2]'))).text
            registered_date = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[3]/div[2]/div[5]/div[1]/div[5]/div/div[2]/div[2]'))).text
            updated_date = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[3]/div[2]/div[5]/div[1]/div[5]/div/div[3]/div[2]'))).text
            print("Will expire on: ", expiry_date)
            expiry_date = "Will expire on: " + expiry_date
            print("Registered on: ", registered_date)
            registered_date = "Registered on: " + registered_date
            print("Updated on:", updated_date)
            updated_date = "Updated on: " + updated_date
            return expiry_date, registered_date, updated_date
        except:
            count += 1
            return domain_info(url, count)
        finally:
            if opendriver:
                driver.close()
    print("No registry information found")

    return ": 0000-00-00", ": 0000-00-00", ": 0000-00-00"


# returns 0 for whitelisted websites
def black_listed(url, count = 0):
    opendriver = False
    while(count < MAX_TRIES):
        try:
            if SCRAPER == "chrome":
                driver = webdriver.Chrome()
            else:
                driver = webdriver.Firefox()
            #driver.minimize_window()
            opendriver = True
            driver.get("https://www.phishtank.com/")
            assert "PhishTank | Join the fight against phishing" in driver.title
            wait = WebDriverWait(driver, 10)
            elem = wait.until(EC.element_to_be_clickable((By.NAME, 'isaphishurl')))
            elem.clear()
            elem.send_keys(url)
            elem.send_keys(Keys.RETURN)
            #assert "PhishTank | Join the fight against phishing" in driver.title
            text = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[2]/div[2]/div/div[1]/div/div/div[2]/form/p/b'))).text
            if text[:19] == "Nothing known about":
                print("This site is not blacklisted")
                return "This site is: not blacklisted"
            else:
                print("This site is blacklisted")
                return "This site is: blacklisted"
        except:
            count += 1
            return black_listed(url, count)
        finally:
            if opendriver:
                driver.close()
    print("Could not find blacklist info")
    return "This site is: Error"


#runs the url against blacklist sources
def reputation_checker(url, count = 0):
    opendriver = False
    while count < MAX_TRIES:
        try:
            if SCRAPER == "chrome":
                driver = webdriver.Chrome()
            else:
                driver = webdriver.Firefox()
            #driver.minimize_window()
            opendriver = True
            driver.get("https://www.urlvoid.com/")
            assert "Check if a Website is Malicious/Scam or Safe/Legit | URLVoid" in driver.title
            wait = WebDriverWait(driver, 10)
            elem = wait.until(EC.element_to_be_clickable((By.ID, 'hf-domain')))
            elem.clear()
            elem.send_keys(url)
            elem.send_keys(Keys.RETURN)
            #assert "Check if a Website is Malicious/Scam or Safe/Legit | URLVoid" in driver.title
            BL_score = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[3]/div[2]/div[2]/div/table/tbody/tr[3]/td[2]/span'))).text
            server_location = wait.until(EC.element_to_be_clickable(
                (By.XPATH, '/html/body/div[3]/div[2]/div[2]/div/table/tbody/tr[9]/td[2]'))).text
            driver.close()
            opendriver = False
            print("Blacklist score: ", BL_score)
            BL_score = "Blacklist score: " + BL_score
            print("Server Location: ", server_location)
            server_location = "Server Location: " + server_location
            return BL_score, server_location
        except:
            count += 1
            return reputation_checker(url, count)
        finally:
            if opendriver:
                driver.close()
    print("Could not find blacklist score or server location")
    return "Blacklist score: Error", "Server Location: Error"


def page_rank(url, count = 0):
    opendriver = False
    while count < MAX_TRIES:
        try:
            if SCRAPER == "chrome":
                driver = webdriver.Chrome()
            else:
                driver = webdriver.Firefox()
            #driver.minimize_window()
            opendriver = True
            driver.get("https://dnschecker.org/pagerank.php")
            assert "Page Rank Checker - Check Your Website Pagerank" in driver.title
            wait = WebDriverWait(driver, 10)
            elem = wait.until(EC.element_to_be_clickable((By.ID, 'prc_host')))
            elem.clear()
            elem.send_keys(url)
            elem.send_keys(Keys.RETURN)
            #assert "Page Rank Checker - Check Your Website Pagerank" in driver.title
            try :
                text = wait.until(EC.element_to_be_clickable(
                    (By.XPATH, '/html/body/div[5]/div/div/div[1]/ul/li[2]/table/tbody/tr/td/span/span'))).text
                print("Page Rank score is : ", text)
                text = "Page Rank score is : " + text
                return text
            except:
                print("Page Rank score is : No pagerank found")
                return "Page Rank score is : No pagerank found"
            finally:
                if opendriver:
                    driver.close()
                    opendriver = False
        except:
            count += 1
            return page_rank(url, count)
        finally:
            if opendriver:
                driver.close()
    print("Could not find pagerank")
    return "Page Rank score is : Error"


def create_url_file(attributes, url, path, addr):
    filename = os.path.join(path, addr + ".txt")
    f = open(filename, "w+")
    time = datetime.datetime.today()
    f.write("URL: " + url + "\n")
    f.write("Report collected on: " + str(time) + "\n")
    for attribute in attributes:
        f.write(attribute + "\n")
    f.close()
    return filename


def debug_path():
    cd = os.getcwd()
    ml = os.path.join(cd, "ML")
    db = os.path.join(ml, "Debug")
    return db


def create_files(urls, path, urlvoid, repeat = True):
    if not repeat:
        completed = os.listdir(path)
        for i in range(len(completed)):
            completed[i] = completed[i][4:-4]
        remove = []
        for url in urls:
            # account for phishing urls
            formatted = format_url(url)
            addr = formatted[0][1]
            addr = addr[4:]
            if addr in completed and url not in remove:
                remove.append(url)
            if url in completed and url not in remove:
                remove.append(url)
        for url in remove:
            urls.remove(url)
    print("Checking {} urls".format(str(len(urls))))
    for url in urls:
        attributes = url_data_collect(url, urlvoid)
        f = create_url_file(attributes, url, path, addr)


def url_data_collect(url, fullurl = None, urlvoid = True):
    if fullurl == None:
        fullurl = url
    print("\n\nInformation from the Data Collector for :", url)
    expiry_date, registered_date, updated_date = domain_info(url)
    bl = black_listed(fullurl)
    if urlvoid:
        BL_score, server_location = reputation_checker(url)
    else:
        BL_score = "Blacklist score: Nothing found"
        server_location = "Server Location: "
    pr = page_rank(url)
    https, cert, cert_chain, hostname, addr = check_all(url)
    attributes = [expiry_date, registered_date, updated_date, bl, BL_score, server_location,
                  pr, https, cert, cert_chain, hostname]
    return attributes


def retrieve_urls(filepath):
    f = open(filepath, "r+")
    content = f.readlines()
    f.seek(0)
    urls = []
    for line in content:
        line = line.strip("\n")
        if line not in urls:
            f.write(line + "\n")
            urls.append(line)
    f.truncate()
    f.close()
    return urls


def small_test_legitimate(urlvoid = True, repeat = True):
    url_path = os.path.join(os.getcwd(), "ML", "legitimate_ecommerce_sites.txt")
    urls = retrieve_urls(url_path)
    urls = urls[:3]
    legit_folder = os.path.join(os.getcwd(), "ML", "0")
    create_files(urls, legit_folder, urlvoid = urlvoid, repeat = repeat)


def debug_test(urlvoid = True, repeat = True):
    urls = ["www.platnosc-payu24.com", "http://sulphureaminhagichun.quickwebchecker.com",
            "https://remit-payoutday.eu/remittance.jnlp",
            "https://yesinfoz.site/chh/ch.php"]
    urls = urls[:3]
    legit_folder = os.path.join(os.getcwd(), "ML", "Debug")
    create_files(urls, legit_folder, urlvoid = urlvoid, repeat = repeat)


def create_all_legit_files(urlvoid = True, repeat = True):
    url_path = os.path.join(os.getcwd(), "ML", "legitimate_ecommerce_sites.txt")
    urls = retrieve_urls(url_path)
    #urls = urls[:3]
    legit_folder = os.path.join(os.getcwd(), "ML", "0")
    create_files(urls, legit_folder, urlvoid = urlvoid, repeat = repeat)


def create_phishing_files(urlvoid = True, repeat = True):
    url_path = os.path.join(os.getcwd(), "ML", "phishing_ecommerce_sites.txt")
    urls = retrieve_urls(url_path)
    # urls = urls[:3]
    phishing_folder = os.path.join(os.getcwd(), "ML", "1")
    create_files(urls, phishing_folder, urlvoid = urlvoid, repeat= repeat)


def format_urls_from_txt(file):
    f = open(file, "r+")
    content = f.readlines()
    f.seek(0)
    urls = []
    for line in content:
        line = line.strip("\n")
        start = line.find("/")
        end = line.find("\t")
        url = line[start + 2:end]
        # form_url = format_url(url)
        # addr = form_url[0][1]
        # addr = addr[4:]
        # if addr not in urls:
        #     f.write(addr + "\n")
        #     urls.append(addr)
        if url[0:4] == "www.":
            url = url[4:]
        if url not in urls:
            f.write(url + "\n")
            urls.append(url)
        f.truncate()
    f.close()


def stage_openphish():
    path = os.path.join(os.getcwd(), "ML", "openphish_staging.txt")
    format_urls_from_txt(path)


def retry_urlvoid():
    legit_folder = os.path.join(os.getcwd(), "ML", "0")
    urls = get_samples_with_urlvoid_fail(legit_folder)
    urls = urls[:30]
    create_files(urls, legit_folder, urlvoid=True, repeat=True)

