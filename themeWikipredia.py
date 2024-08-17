# -*- coding: utf-8 -*-
# import requests
# from bs4 import BeautifulSoup
# import re
# from collections import defaultdict
import pickle
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer
import os

# with open('theme_dict.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

# keys = loaded_dict.keys()
# print(keys)

# User-Agent 설정
# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
#     'Accept-Language': 'ko-KR,ko;q=0.9',
#     'Accept-Encoding': 'gzip, deflate, br',
#     'Connection': 'keep-alive'
# }

# print(f"********************  {query}  ********************")
# url = f'https://ko.wikipedia.org/wiki/{query}'
# response = requests.get(url, headers=headers)
# soup = BeautifulSoup(response.text, 'html.parser')
# elements = soup.find_all(class_='mw-content-ltr mw-parser-output')
# text = ""
# for element in elements:
#   text = element.get_text()

# text = text.split("같이 보기")[0]

# okt = Okt()
# nouns = okt.nouns(text)

# text = ' '.join(nouns)

# print(text)

title ="수산"
text = """
"수산"이라는 용어는 일반적으로 수산업, 즉 어업과 해양 생물 자원을 다루는 산업을 의미합니다. 수산업은 어획, 양식, 해양 생물의 가공 및 유통 등을 포함하며, 인간의 식량 공급과 경제 활동에 중요한 역할을 합니다. 아래에서 수산업의 주요 개념, 종류, 특징 및 현재 동향에 대해 설명하겠습니다.

1. 수산업의 주요 개념
어업: 바다나 강, 호수에서 물고기 및 기타 해양 생물을 잡는 활동입니다.
양식: 인공적으로 물고기, 조개, 갑각류 등을 기르는 활동으로, 자연 생태계에서의 자원 고갈을 방지하는 데 기여합니다.
수산물 가공: 잡은 해양 생물을 가공하여 소비자에게 판매하는 과정으로, 신선한 상태에서 냉동, 건조, 통조림 등 다양한 형태로 가공됩니다.
2. 수산업의 종류
상업 어업: 대규모로 해양 생물을 잡아 판매하는 어업 형태로, 산업적으로 운영됩니다.
소형 어업: 지역 주민들이 생계를 위해 소규모로 어획하는 형태입니다.
양식업: 해양, 담수에서 인공적으로 생물을 기르는 산업으로, 다양한 품종의 생물을 양식할 수 있습니다.
3. 특징
자원 관리: 지속 가능한 수산업을 위해 어획량과 생태계를 관리하는 것이 중요합니다.
기후 변화의 영향: 해양 생물은 기후 변화에 민감하여, 수산업도 이에 따라 영향을 받을 수 있습니다.
수출 산업: 수산물은 많은 국가에서 중요한 수출 품목으로, 경제적 가치가 큽니다.
4. 현재 동향
지속 가능성: 환경 보호와 지속 가능한 자원 관리를 위한 노력이 증가하고 있으며, 친환경 양식과 어업 방식이 주목받고 있습니다.
기술 발전: 자동화, AI, 데이터 분석 등을 활용한 스마트 어업 기술이 발전하고 있습니다.
해양 생태계 보호: 오염, 남획 등으로 인한 해양 생태계 파괴를 방지하기 위한 정책과 국제 협력이 필요합니다.
수산업은 인류의 중요한 식량 자원과 경제적 기반을 제공하는 분야로, 지속 가능한 발전과 환경 보호를 위해 다양한 노력이 필요합니다.
"""
okt = Okt()
nouns = okt.nouns(text)
text = ' '.join(nouns)
print(text)

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
text_embedding = model.encode([text])

print(text_embedding)

with open(f'C:/Topic_Modeling/embeddings/{title}_embedding.pkl', 'wb') as f:
    pickle.dump(text_embedding, f)
