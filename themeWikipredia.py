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
error_list = []
query_list = ['모더나', '진단키트', '메르스', '스푸트니크V', '음압병실', '마스크', '황사', '반도체', 'mRNA', 'HBM', '페라이트', '미용기기', '코로나19', '선박평형수', '낸드', '마이코플라스마', '바이오시밀러', '빈대', '유전자 치료', '보톡스', '제지', '치매', '출산장려정책', '화장품', '백신', '온디바이스 AI', '뉴로모픽', '면역항암제', 'SSD', 'IT', 'K-뉴딜지수(바이오)', 'MVNO', '혈장치료', '공기청정기', '의료AI', 'CXL', '제약', '마리화나', 'LED', '전력', '전선', '줄기세포', '일자리', '반도체', '셰일', 'PCB', '덱사메타손', '종합상사', '불매운동', '원격진료', '의료기기', '로봇', '조림사업', '콜드체인', '자동차', '희귀금속', '올림픽','월드컵', 'LCD', '타이어', '아이폰', '테마파크', '탈모', '패션', '보안주', '2차전지', 'NI', '전기자전거', '스마트그리드', 'SNS', '시스템반도체', '2020 하반기 신규상장', '3D 프린터', '카모스타트', '골프', '교육', '무선충전', '유리 기판', '주류', '캐릭터', '스마트홈', '반도체', '슈퍼박테리아', '폐배터리', '고령화', '2022 하반기 신규상장', '온실가스', '농업', '야놀자', 'SI', '카카오뱅크', '겨울', '마이크로바이옴', '플렉서블 디스플레이', '2021 하반기 신규상장', '제습기', '재택근무', '강관업체', '2019 상반기 신규상장', '인터넷은행', '두나무(Dunamu)', 'CCTV', 'SOFC', '제대혈', '니켈', 'LED', 'DMZ', '자동차부품', '공작기계', '자율주행차', '엔젤산업', '화학섬유', '제4이동통신', '철강', '우주항공', '그래핀', '음원', '수소차', '전자파', '日 수출 규제(국산화 등)', '리튬', '리비안', 'NFT', '홈쇼핑', 'LCD 부품/소재', '철도', '스마트폰', '구충제', '카지노', '폴더블폰', '전기차', '2023 하반기 신규상장', '블랙박스', '면세점', '보안주', '국내 상장 중국기업', 'ESS', '철강', '나파모스타트', '휴대폰부품', '조선기자재', '지주사', '네옴시티', 'OLED', '음성인식', '2021 상반기 신규상장', 'MRO', '영화', '마켓컬리', '삼성페이', '관이음쇠', '인테리어', '비료', '블록체인', '냉각시스템', '치아 치료', '비철금속', 'STO', '클라우드', '건강기능식품', '2023 상반기 신규상장', '스마트팩토리', '풍력에너지', 'LCD BLU제조', '백신여권', '방위', '가스관사업', '건설', '2022 상반기 신규상장', '2024 상반기 신규상장', '항공', '태블릿PC', '원자력발전', '2차전지(소재/부품)', '은행', '2024 하반기 신규상장', '가상화폐', '챗봇', 'UAM', 'AI', '핵융합에너지', '마이데이터', '증권', '요소수', '손해보험', '사물인터넷', '수자원', '마이크로', '맥신', '기업인수목적회사', '원자력발전소 해체', '모듈러주택', '석유화학', '화폐', '렌터카', 'REITs', '2019 하반기 신규상장', 'LNG', 'K-뉴딜지수', '갤럭시 부품주', '태양광에너지', '2차전지', '창투사', '핀테크', '폐기물처리', '양자암호', '전자결제', '시멘트', '스마트폰', '항공기부품', '가상현실', '비만치료제', '인터넷', '쿠팡', '모바일콘텐츠', '생명보험', 'K-뉴딜지수(2차전지)', '광고', '크래프톤', '키오스크', '터치패널', '5G', '소매유통', '해저터널', '미디어', '4차산업', '태풍', '장마', '메타버스', '통신장비', '종합 물류', '전기차(충전소/충전기)', '스마트카', '건설기계', '여행', '바이오인식', '자전거', '남북경협', '2차전지(전고체)', '2020 상반기 신규상장', 'MLCC', '백화점', '골판지', '통신', '편의점', '조선', '2차전지(생산)', '카메라', '밥솥', '탄소나노튜브', 'GTX', '영상콘텐츠', '여름', '아프리카 돼지열병', '자원개발', '엔터테인먼트', '정유', '전기차 화재 방지', '페인트', '2차전지', '재난', '환율하락', '게임', '도시가스', '북한 광물자원개발', '애플페이', '증강현실', '모바일게임', '드론', '우크라이나', '육계', '윤활유', '건설 중소형', '음식료', 'RFID', '구제역', '광우병', '4대강', '해운', '사료', '웹툰', 'K-뉴딜지수(인터넷)', 'K-뉴딜지수(게임)', '초전도체', 'LPG', '케이블TV', '아스콘', '수산']
for query in query_list:
  print(f"********************  {query}  ********************")
  url = f'https://ko.wikipedia.org/wiki/{query}'
  response = requests.get(url, headers=headers)
  soup = BeautifulSoup(response.text, 'html.parser')
  elements = soup.find_all(class_='mw-content-ltr mw-parser-output')
  text = ""
  for element in elements:
    text = element.get_text()
  if len(text) > 100:
    text = text.split("같이 보기")[0]

    okt = Okt()
    nouns = okt.nouns(text)

    text = ' '.join(nouns)

    print(text)

    model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
    text_embedding = model.encode([text])

    with open(f'./embeddings/{query}_embedding.pkl', 'wb') as f:
        pickle.dump(text_embedding, f)
    print(text_embedding)
  else:
    error_list.append(query)
    print(f"'{query}' 오류 발생: 다음 쿼리로 넘어갑니다.")
print(error_list)