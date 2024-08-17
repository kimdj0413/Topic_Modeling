# -*- coding: utf-8 -*-
import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import pickle

##      뉴스기사
article = """
16일 국내 최대 중고차 거래 플랫폼 엔카에 따르면 이날 기준 이 업체 웹사이트에 매매 등록된 메르세데스-벤츠 EQE 350+모델은 총 85대다. 이들 차량은 지난달만 해도 연식(2022년~2024년식)에 따라 6000만대 중반부터 7000만원대까지 중고 시세가 형성했는데 청라 화재 사고 이후에는 6000만원대 초반으로 급락했다. 이주 들어서는 급기야 5900만원대를 깬 5850만원짜리 매물도 등장했다.

업계 한 관계자는 “통상 자동차 감가상각은 신차 출고 이후 1년~3년 사이에 높게 이뤄지고, 감가 폭은 연 10% 규모”라며 “특히 전기차는 내연기관차에 비해 시중에 풀린 물량이 많지 않아 감가가 더 크게 이뤄지고 시세도 수요에 따라 요동치는 편인데 이번에 전기차 화재 사고로 가격 하락폭이 더 커졌다”고 설명했다. 벤츠 EQE 350+ 신차 출고가가 1억380만원인 점을 고려하면 2022년식 차량이 2년 만에 40％ 넘게 가격이 빠졌다.

화재 사고 이후 중고차 시장에선 벤츠 전기차뿐만 아니라 다른 브랜드의 전기차 차주들도 차량을 처분하려는 움직임이 늘고 있다. 국내 최대 직영중고차 업체 케이카(K Car)에 따르면 지난주(1~7일) ‘내차 팔기 홈 서비스’에 등록된 전기차 접수량은 직전 주(지난달 25∼31일) 대비 184% 급증했다. 차량 대수는 공개하지 않았지만, 이중 벤츠 EQE 시리즈 모델이 10% 정도 차지했다. 케이카 관계자는 “이주(8~14일) 들어 등록된 접수량은 전주(1~7일) 물량의 90% 수준”이라며 “급격히 떨어진 전기차 가격이 진정될 때까지 기다리려는 이들이 늘면서 매물 접수도 일단 진정된 상태”고 설명했다.

전기차를 처분하려는 이들은 늘었지만, 이번 화재 사고로 중고차 업계에선 전기차 매입을 꺼려하는 분위기도 빠르게 확산중이다. 서울의 한 중고차 딜러는 벤츠 EQE 모델의 경우 아예 ‘매입 거부’를 공개적으로 밝힌 곳도 등장했다. 상대적으로 가격이 비싼 수입차의 경우 실내 주차장에 보관하는 경우가 많은데, 이번에 화재가 난 차량도 충전 중이 아닌 상태에서 불이 났던 만큼 차량을 보관하는 것도 우려되기 때문이다.

자동차 업계는 가뜩이나 침체한 전기차 시장이 이번 화재 사고 이후 수요 둔화가 더욱 장기화할까 우려하는 분위기다. 업계 관계자는 “완성차업계가 전기차 캐즘(일시적 수요 둔화)에 이어 포비아(공포증)까지 퍼지면서 신차 판매에 당장 제동이 걸린 상황”이라며 “전기차 프로모션과 함께 배터리 정보도 자발적으로 공개하며 소비자 불안을 해소하는 데 총력을 기울이고 있지만 수요를 회복하기엔 쉽지 않은 상황”이라고 말했다.

한편, 국토교통부 자동차리콜센터가 공개한 ‘제작사별 차명별 배터리 제조사’ 현황에 따르면 이날 기준 국내에서 전기차를 판매하는 업체 중 현대차와 기아, 르노코리아, KGM, 벤츠, BMW, 볼보, 스텔란티스, 포르쉐, 폴스타, 폭스바겐, 토요타, 테슬라 등 13개 완성차 브랜드가 배터리 제조사 정보를 공개한 상태다.
"""

##      형태소 분석 및 N Gram 생성
okt = Okt()
nouns = okt.nouns(article)
text = ' '.join(nouns)

n_gram_range = (1,2)
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
  
keyword = mmr(doc_embedding, candidate_embeddings, candidates, top_n=3, diversity=0.7)
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

# 특정 디렉토리 경로
directory_path = 'C:/Topic_Modeling/embeddings'

# 디렉토리의 모든 파일 이름 불러오기
doc_embedding_list = []
name_list = []
file_names = os.listdir(directory_path)
for file in file_names:
    name_list.append(file)
    with open(f'C:/Topic_Modeling/embeddings/{file}', 'rb') as file:
        doc_embedding = pickle.load(file)
        doc_embedding_list.append(doc_embedding)

cosine_score=[]
candidate_embeddings = model.encode(keyword)
for doc_embedding in doc_embedding_list:
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
    sum=0
    for sim in word_doc_similarity:
        sum += sim
    cosine_score.append(sum/len(word_doc_similarity))

indexed_numbers = [(value, index) for index, value in enumerate(cosine_score)]
sorted_numbers = sorted(indexed_numbers)
smallest_three = sorted_numbers[10:20]
smallest_values = [value for value, index in smallest_three]
smallest_indices = [index for value, index in smallest_three]

for index in smallest_indices:
    print(name_list[index])