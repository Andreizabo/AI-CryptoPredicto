from time import sleep
import json

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.chrome.options import Options


def get_news():
    base_url = 'https://www.coindesk.com/search'
    coins = ['ethereum', 'litecoin', 'binance%20coin']
    categories = ['Markets', 'Business', 'Tech', 'Policy']

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    browser = webdriver.Chrome(options=options)
    browser.maximize_window()

    results = {}

    for coin in coins:
        results[coin] = []
        for category in categories:
            url = base_url + '?s=' + coin + '&cf=' + category
            browser.get(url)
            page_num = 0
            while True:
                print(f'Scraping page {page_num} from coin {coin} and category {category}')
                WebDriverWait(browser, 10).until(presence_of_element_located((By.CLASS_NAME, "hpnSzW")))
                while True:
                    try:
                        news_divs = browser.find_elements(By.CLASS_NAME, "hpnSzW")[4:]
                        aux = []
                        for news_div in news_divs:
                            news = news_div.find_elements(By.TAG_NAME, "a")[1:3]
                            date = news_div.find_elements(By.TAG_NAME, "h6")[1].text
                            title = news[0].text
                            text = news[1].text if len(news) > 1 else ''
                            aux.append({'date': date, 'title': title, 'text': text})
                        results[coin].extend(aux)
                        break
                    except Exception as e:
                        print('Am avut Ion Buleala :(')
                        print(e)
                        continue

                try:
                    next_button = browser.find_elements(By.CLASS_NAME, "kdxKRn")[0 if page_num == 0 else 1]
                    browser.execute_script("arguments[0].click();", next_button)
                    # sleep(20)
                except Exception as e:
                    print(e)
                    sleep(30)
                    break
                page_num += 1

    browser.close()
    return results


if __name__ == '__main__':
    with open('news.json', 'w') as f:
        json.dump(get_news(), f, indent=4)
