import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import pickle

page = 0
theme_list = []
source_list = []

# User-Agent 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'ko-KR,ko;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

theme_dict = defaultdict(list)

for i in range(0, 8):
  page += 1
  url = f'https://finance.naver.com/sise/theme.naver?&page={page}'
  response = requests.get(url, headers=headers)

  if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    theme_elements = soup.find_all(class_='col_type1')
    for theme_element in theme_elements:
      a_tag = theme_element.find('a')
      if a_tag and a_tag.get_text() != '테마명':
        stock_list = []
        theme_name = a_tag.get_text()
        theme_href = a_tag['href']
        source = ('https://finance.naver.com'+theme_href)
        stock_response = requests.get(source, headers=headers)
        stock_soup = BeautifulSoup(stock_response.text, 'html.parser')
        stock_elements = stock_soup.find_all(class_='name_area')
        cnt = 0
        for stock in stock_elements:
            stock_text = stock.get_text()
            cleaned_text = re.sub(r'[^가-힣a-zA-Z0-9]', '', stock_text)
            theme_dict[theme_name].append(cleaned_text)
            cnt += 1
            if cnt == 5:
              break

print(dict(theme_dict))

with open('theme_dict.pkl', 'wb') as f:
    pickle.dump(theme_dict, f)
    
with open('theme_dict.txt', 'w', encoding='utf-8') as f:
    for key, values in theme_dict.items():
        f.write(f"{key}: {', '.join(values)}\n")

with open('theme_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

theme = '황사/미세먼지'
value = ', '.join(loaded_dict[theme])
print(f"'{theme}' 테마 주도주들 : {value}")