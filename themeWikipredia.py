# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import pickle
from konlpy.tag import Okt
from sentence_transformers import SentenceTransformer

# with open('theme_dict.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)

# keys = loaded_dict.keys()
# print(keys)

# User-Agent 설정
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'ko-KR,ko;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

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

title ="코로나19(음압병실/음압구급차)"
text = """
음압병상은 기압 차를 이용해 병실 내부의 공기가 외부로 빠져나가지 못하도록 해, 병원균과 바이러스를 차단하는 기능을 하는 병실이다. 즉 기압 차로 인해 병실 밖의 공기는 들어오지만 병실 안의 공기는 밖으로 나가지 못하도록 설계된 공간이다. 특히 환자의 호흡 등으로 배출된 바이러스가 섞인 공기가 밖으로 나가지 않고 천장 정화 시설로 흐르도록 돼 있다. 따라서 내부 오염원이 정화시설을 통해 걸러지면서 바이러스의 외부 유출이 방지되는 것이다. 

음압병상은 환자를 외부 및 일반 환자들과 분리해 수용·치료하기 위한 특수 격리 병실로, 감염병 확산을 막기 위한 필수시설이다. 음압병실은 크게 전실과 환자가 입원하는 병실 등 2개의 공간으로 구성되는데, 의료진은 전실에서 손 소독과 방호복 착용 뒤 병실로 들어가게 된다. 병실은 중증 환자를 치료할 수 있도록 설계돼 있으며, 감염병 환자와 병원 직원의 동선을 분리하기 위해 출입문은 따로 설치된다.
음파에 의해 생긴 압력의 변화량. 이론적으로 음파의 세기(강도)를 나타내는 양(量)의 하나로, 음파에 의해 생긴 미소 출력의 변화를 말한다. 음파의 세기를 나타내는 다른 양으로 입자 속도가 있다. 음향학에서는 단위 면적을 단위 시간에 흐르는 에너지를 음의 세기라 한다.
"""
okt = Okt()
nouns = okt.nouns(text)
text = ' '.join(nouns)
print(text)

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
text_embedding = model.encode([text])

with open(f'C:/Topic_Modeling/embeddings/{title}_embedding.pkl', 'wb') as f:
    pickle.dump(text_embedding, f)
print(text_embedding)