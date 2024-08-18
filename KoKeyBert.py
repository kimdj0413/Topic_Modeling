# -*- coding: utf-8 -*-
import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

doc = """
        '투자의 귀재' 워런 버핏이 애플 주식 절반을 팔고 화장품 업체 주식을 사들였습니다.

정확히는 미국의 화장품 판매 유통 기업인데, 이 소식이 알려지면서 우리나라 화장품 업체들 주가까지 덩달아 들썩였습니다.

배정현 기자가 보도합니다.

[기자]
'투자의 귀재' 워런 버핏이 이끄는 버크셔 해서웨이가 올 2분기에 애플 지분은 절반 가량 팔고, 미국 화장품 업체인 울타뷰티 주식을 3620억 원 가량 사들였습니다.

울타뷰티는 500개 이상의 브랜드가 입점해 있는 미국의 유명 화장품 판매 유통업체로, 버핏이 해당 업체의 가치가 저평가됐다고 본 것 아니냐는 해석이 나옵니다.

버핏이 매수한 해당 화장품 업체에는 한국 기업 제품들도 다수 입점해 있는데요.
 
매수 소식에 한국 화장품업체들 주가도 일제히 올랐습니다.

울타뷰티에 입점돼 있는 토니모리와 마녀공장 등은 전 영업일보다 7% 넘게 올랐습니다.

화장품 제조업체인 한국화장품제조도 한때 최고가를 경신하며 오늘 하루에만 20% 이상 올랐습니다.

버핏의 선택이 화장품의 견조한 수요에 대한 시그널이 된 겁니다.

[김혜미 / 상상인증권 연구원]
"최근에 우리나라 화장품 업체들이, 미국 진출 모멘텀으로 많이 실적도 개선하고 주가도 좋다 보니까 그게 좀 연동이 돼서 (업계에서는) 호재로 인식을 하는 것 같고요."

해외에서 K-뷰티가 각광을 받고 있는 만큼 국내 화장품 업계 성장에 대한 기대감도 커지고 있습니다.
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
# print(candidates)
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