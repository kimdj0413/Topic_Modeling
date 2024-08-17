# -*- coding: utf-8 -*-
import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import pickle
import re

##      뉴스기사
article = """
정부가 내수가 살아날 조짐을 보인다는 경기 진단을 넉 달째 이어갔다. 길어지는 내수 부진을 이유로 국책 연구기관은 올해 성장률 전망치까지 낮춘 만큼 내수 회복세를 낙관하긴 이르다는 지적이 나온다.

기획재정부는 16일 ‘최근 경제동향(그린북)’ 8월호에서 “최근 우리 경제는 견조한 수출·제조업 호조세에 설비투자 중심의 완만한 내수 회복 조짐을 보이며 경기 회복 흐름이 지속되는 모습”이라고 평가했다. 올 5월 ‘내수 회복 조짐’을 처음 꺼낸 정부가 넉 달째 이런 시각을 유지한 것이다.

김귀범 기재부 경제분석과장은 “실질임금이 두 달 연속 상승했고 방한 관광객과 카드 매출액 속보 지표도 연초 이후 개선 흐름을 보이고 있다. 소비자심리지수도 2020년 4월 이후 가장 높은 수준”이라고 설명했다.

하지만 정부의 이런 시각은 한국개발연구원(KDI) 등 외부의 시각과는 온도 차가 있다. KDI는 앞서 7일 ‘경제전망 수정’에서 올해 성장률 전망치를 2.6%에서 2.5%로 0.1%포인트 낮춰 잡았다. 당초 예상보다 내수가 미약한 수준에 그치면서 경기 회복의 발목을 잡고 있다는 판단이었다. KDI의 전망치는 기재부가 6월 전망한 올해 성장률(2.6%)보다 낮다.

실제 내수 상황을 보여주는 일부 지표는 부진을 벗어나지 못하고 있다. 상품 소비를 보여주는 소매판매는 1년 전보다 3.6% 감소했다. 최근에는 특히 백화점과 할인점의 매출이 줄줄이 쪼그라들면서 소비 지표가 꺾이고 있다. 지난달 백화점의 국내 카드 승인액은 1년 전보다 1.4% 줄며 두 달 연속 감소세를 이어갔고, 할인점 매출액은 3.3% 감소해 6월보다 감소 폭이 더 커졌다. 설비투자 역시 1년 전보다 2.7% 줄었다.

7월 물가상승률 역시 6월(2.4%)보다 소폭 확대된 2.6%였다. 집중호우와 유가 상승 등의 영향으로 농산물, 석유류 물가가 비싸진 탓이다. 건설경기 악화에 건설투자 역시 4.6% 줄었다. 이 때문에 정부가 내놓은 내수 진단이 서민들이 느끼는 체감 경기를 반영하지 못한다는 지적도 나온다.

하준경 한양대 경제학부 교수는 “물가가 전보다 안정되긴 했지만 최근 2년간 실질소득이 뒷걸음질한 상황이라 내수 회복의 여건이 되지 못하고 있다”며 “내수 회복세를 낙관하기는 이르다”고 말했다. 그는 “내수가 1분기(1∼3월) 반짝 회복세를 보인 건 정부가 재정을 당겨 쓴 영향도 있다. 이제는 그럴 재정도 남아 있지 않다”고 덧붙였다.
"""

##      형태소 분석 및 N Gram 생성
okt = Okt()
nouns = okt.nouns(article)
text = ' '.join(nouns)

n_gram_range = (2,3)
count = CountVectorizer(ngram_range=n_gram_range).fit([text])
candidates = count.get_feature_names_out()

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding = model.encode([text])
candidate_embeddings = model.encode(candidates)

##    MMR 기법
def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
    word_similarity = cosine_similarity(candidate_embeddings)
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]
  
keyword = mmr(doc_embedding, candidate_embeddings, candidates, top_n=3, diversity=0.1)
print(keyword)

# # 특정 디렉토리 경로
# directory_path = 'C:/Topic_Modeling/embeddings'

# # 디렉토리의 모든 파일 이름 불러오기
# file_names = os.listdir(directory_path)
# name_list = []

# # 파일 이름 리스트
# for file in file_names:
#     name_list.append(file)
# name_list = [name.replace('_embedding.pkl', '') for name in name_list]
# print(name_list)

# # 특정 디렉토리 경로
# directory_path = 'C:/Topic_Modeling/embeddings'

# # 디렉토리의 모든 파일 이름 불러오기
# doc_embedding_list = []
# name_list = []
# file_names = os.listdir(directory_path)
# for file in file_names:
#     name_list.append(file)

#     with open(f'C:/Topic_Modeling/embeddings/{file}', 'rb') as file:
#         doc_embedding = pickle.load(file)
#         doc_embedding_list.append(doc_embedding)

# cosine_score=[]
# candidate_embeddings = model.encode(keyword)
# for doc_embedding in doc_embedding_list:
#     word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
#     sum=0
#     for sim in word_doc_similarity:
#         sum += sim
#     cosine_score.append(sum/len(word_doc_similarity))

# indexed_numbers = [(value, index) for index, value in enumerate(cosine_score)]
# sorted_numbers = sorted(indexed_numbers)
# smallest_three = sorted_numbers[10:20]
# smallest_values = [value for value, index in smallest_three]
# smallest_indices = [index for value, index in smallest_three]

# for index in smallest_indices:
#     print(name_list[index])

with open('C:/Topic_Modeling/theme_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

keys = loaded_dict.keys()
keys_list = list(keys)

new_keys_list = [re.sub(r'[^a-zA-Z0-9가-힣\s]', ' ', key) for key in keys_list]

key_embeddings = model.encode(new_keys_list)
print(mmr(doc_embedding, key_embeddings, keys_list, top_n=3, diversity=0.1))