# -*- coding: utf-8 -*-
import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

doc = """
        모빌리티 플랫폼 쏘카가 적자를 기록했다. 체질 개선을 위한 투자를 우선했기 때문이다. 하반기부터는 흑자 전환을 위한 움직임을 본격화하겠다는 구상이다. 

쏘카는 올해 2분기 연결 기준 영업손실이 약 67억원으로 잠정 집계됐다고 13일 공시했다. 지난해 2분기 영업이익은 16억원이었으나 이번에는 적자를 냈다.

2분기 매출액은 약 1018억원으로 전년 동기 1039억원 대비 약 2.0% 감소했다. 당기순손실은 약 116억원을 기록했다.

쏘카는 차량 이용주기 확대를 위해 중고차량 매각을 최소화하면서 특히 2분기 중고차 판매 매출이 지난해 172억원에서 올해 10억원으로 약 94.2% 감소했다고 설명했다. 아울러 카셰어링 부문 수요 창출, 플랫폼 부문 투자, 마케팅 확대로 간접비가 1년 새 약 33.4% 늘면서 영업 적자 요인으로 작용했다.

단기 카셰어링과 쏘카플랜(장기)으로 이루어진 카셰어링 부문에서는 2분기에만 914억원의 매출을 기록했다. 특히 단기 카세어링의 매출총이익률(GPM)은 지난해 2분기 17.1%에서 올해 18.9%로 올랐다. 

운1개월 이상 대여상품 쏘카플랜 GPM은 지난해 2분기 26.7%에서 올해 마이너스(-)8.1%로 급감했다. 상반기 운영 차량을 3배 확대하며 공격적인 프로모션을 전개한 영향이 컸다. 다만 올해 1분기와 견주면 7.2%포인트 상승한 수치다. 

쏘카는 쏘카플랜 GPM이 지난 6월부터 반등해 3분기부터는 다시 분기 전체 '플러스(+)'로 전환할 것으로 전망하고 있다. 차량·이용자 생애주기이익(LTV) 증대를 위해 지난해 하반기부터 시작한 쏘카 2.0 전략이 본궤도에 진입하며 매출총이익률도 본격 개선세에 접어들었다는 설명이다.

공유 전기자전거 '일레클', 온라인 주차 플랫폼 '모두의주차장', 숙박 등 플랫폼 부문 총거래액(GMV)은 서비스 라인업 확장에 따라 2분기 255억원을 기록했다. 전년 동기 대비 약 27.2%(55억원) 증가했다. 일레클의 일회성 기기판매 거래액을 제외하면 약 47.8% 늘었다. 같은 기간 월간 앱 방문자수(MUV)는 약 1.2배 증가한 152만명을 기록했다.

쏘카는 부름·편도 등 고부가 서비스 확대, 네이버 채널링 효과 등으로 3분기 흑자 달성을 노리고 있다. 2분기까지 쏘카 2.0 전략을 위한 전략적 투자가 일단락되고 하반기부터는 마케팅비용도 줄어들어 안정적인 수익 달성에 기여할 것이라고 설명했다.

박재욱 쏘카 대표는 "지난 1년간 전개해 온 쏘카 2.0 전략을 통해 카셰어링 시장의 성장 잠재력과 시장을 리드하는 쏘카의 능력을 입증했다"며 "3분기부터 안정적으로 이익을 창출할 수 있는 전사 차원의 체질 개선을 확신한다"고 말했다.
      """
split_doc = re.split(r'(\s*[a-zA-Z]+(?:[^\w\s]+[a-zA-Z]+)*\s*)', doc)

# 불필요한 공백 제거
split_doc = [part.strip() for part in split_doc if part.strip()]

##    형태소 분석
okt = Okt()
nouns=[]
for sentence in split_doc:
    sen = okt.nouns(sentence)
    if len(sen) != 0:
        nouns.extend(sen)
    else:
        nouns.append(sentence)

text = ' '.join(nouns)

##    N Gram 생성
n_gram_range = (1,2)

count = CountVectorizer(ngram_range=n_gram_range).fit([text])
candidates = count.get_feature_names_out()
print(candidates)
##    SBERT 사용해 수치화
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

##    상위 5개 출력
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)

def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]
print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10))
print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=30))
# ##    MMR 기법 사용해 다양성 최대화
# def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

#     # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
#     word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

#     # 각 키워드들 간의 유사도
#     word_similarity = cosine_similarity(candidate_embeddings)
#     print(f'word_similarity : {word_similarity}')

#     # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
#     # 만약, 2번 문서가 가장 유사도가 높았다면
#     # keywords_idx = [2]
#     keywords_idx = [np.argmax(word_doc_similarity)]

#     # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
#     # 만약, 2번 문서가 가장 유사도가 높았다면
#     # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
#     candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

#     # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
#     # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
#     for _ in range(top_n - 1):
#         candidate_similarities = word_doc_similarity[candidates_idx, :]
#         target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

#         # MMR을 계산
#         mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
#         mmr_idx = candidates_idx[np.argmax(mmr)]

#         # keywords & candidates를 업데이트
#         keywords_idx.append(mmr_idx)
#         candidates_idx.remove(mmr_idx)

#     return [words[idx] for idx in keywords_idx]
  
# print(mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.5))