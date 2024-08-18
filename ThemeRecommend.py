# -*- coding: utf-8 -*-
import numpy as np
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import pickle
import re
import itertools

##      뉴스기사
article = """
인공지능(AI) 스타트업 사카나AI는 지난 12일 ‘AI 사이언티스트’를 공개했다. 대규모언어모델(LLM)을 활용해 과학 관련 연구를 AI가 독자적으로 진행하는 솔루션이다. AI 사이언티스트는 아이디어 창출, 실험 수행, 결과 요약, 논문 작성, 리뷰까지 과학 연구의 모든 과정을 자동으로 처리한다. 업계에선 “AI가 스스로 더 나은 버전을 개발하는 단계에 이르렀다”는 평가가 나왔다. AI 불모지로 여겨지던 일본에서 나온 기업이란 것도 사카나AI가 주목받는 이유 중 하나다.
18일 업계에 따르며 최근 일본 정부는 자국 AI 스타트업 지원을 대폭 강화해 AI산업을 키우고 있다. 최근엔 창업 2년 차 AI 스타트업을 유니콘 기업(기업가치 10억달러 이상 비상장사)으로 키우는 데 성공했다.

스타트업 분석업체 CB인사이츠에 따르면 올 2분기 유망 AI 스타트업 여섯 곳이 유니콘 기업 반열에 새로 올랐다. 5개는 미국 기업으로 xAI, 자이라테라퓨틱스, 코그니션AI 등이다. 나머지 한 곳이 일본의 LLM 스타트업 사카나AI다.

사카나AI는 지난해 7월 설립돼 창업 1년도 되지 않아 유니콘 기업에 올랐다. 미국의 유명 벤처캐피털(VC)인 럭스캐피털, 코슬라벤처스 등이 투자했다. NTT, KDDI, 소니 등 일본의 정보기술(IT) 대기업도 투자에 참여했다. 제프 딘 구글 수석과학자, 클레망 델랑지 허깅페이스 창업자 등 글로벌 AI업계 유명 인사도 시드(초기) 단계부터 투자에 나섰다.

사카나AI의 경쟁력은 우수 인력이다. 구글의 핵심 AI 연구원 출신인 라이언 존스가 공동 창업자로 참여했다. 존스는 최근 AI 혁신의 바탕인 ‘트랜스포머’라는 AI 알고리즘을 처음 제시한 논문 ‘어텐션 이즈 올 유 니드(Attention is all you need)’의 저자 중 한 명이다. 데이비드 하, 로버트 랑케 등 다른 공동 창업 멤버도 구글 출신의 외국인이다. 일본인 공동 창업자는 일본 중고 거래 플랫폼업체 메루카리의 유럽지사장을 지낸 렌 이토 등이다. 사카나라는 회사명은 ‘물고기’를 뜻하는 일본어에서 따왔다.
○일본 정부의 아낌없는 지원
일본 정부의 전폭적인 지원도 사카나AI가 고속 성장할 수 있었던 배경이다. 일본은 자국 AI산업 경쟁력을 강화하기 위해 해외 인재 유치에 적극적으로 나서고 있다.

우선 외국인 창업 규제를 완화했다. 사무실, 출자금 등의 조건 없이 사업 계획이 인정되면 2년간 체류할 수 있도록 비자 요건을 지난해 낮췄다. 기존에는 외국인이 일본에서 사업하기 위해선 사무실과 2명 이상의 상근 직원, 500만엔(약 4600만원) 이상의 출자금 등을 갖춰야 했다. 지난해 ‘특별고도인재’ 비자도 신설해 해외 인재에게 5년짜리 비자를 바로 내주고 있다.

일본은 전 세계적으로 품귀현상을 빚고 있는 고성능 그래픽처리장치(GPU)도 정부 차원에서 확보해 무상으로 지원하고 있다. 사카나AI도 일본 정부의 ‘생성형 AI 개발 지원 프로그램(GENIAC)’을 통해 GPU 문제를 해결했다. 엔비디아의 H100 등 고성능 GPU는 AI 기술 개발에 필수적이지만, 관련 인프라 확보에 수백억원이 들어간다. 초기 스타트업이 자체 예산으로 구비하는 게 불가능한 구조다.

한국에는 아직 AI 유니콘 기업이 없다. 국내총생산(GDP) 규모가 한국보다 작은 이스라엘, 싱가포르 등에도 AI 유니콘 기업이 있는 것과 대조적이다. AI 인재도 유입보다 유출이 많다.

미국 스탠퍼드대 인간중심AI연구소(HAI)에 따르면 지난해 한국은 인도와 이스라엘에 이어 AI 인재 유출이 세 번째로 많은 국가였다.
"""
split_article = re.split(r'(\s*[a-zA-Z]+(?:[^\w\s]+[a-zA-Z]+)*\s*)', article)

# 불필요한 공백 제거
split_article = [part.strip() for part in split_article if part.strip()]

##      형태소 분석 및 N Gram 생성
okt = Okt()
nouns=[]
for sentence in split_article:
    sen = okt.nouns(sentence)
    if len(sen) != 0:
        nouns.extend(sen)
    else:
        nouns.append(sentence)

text = ' '.join(nouns)

n_gram_range = (1,2)
count = CountVectorizer(ngram_range=n_gram_range).fit([text])
candidates = count.get_feature_names_out()

model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding = model.encode([article])
candidate_embeddings = model.encode(candidates)

def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates):
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [words[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]
print(max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=5, nr_candidates=10))

##  테마 추천
with open('C:/Topic_Modeling/theme_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

keys = loaded_dict.keys()
keys_list = list(keys)

new_keys_list = [re.sub(r'[^a-zA-Z0-9가-힣\s]', ' ', key) for key in keys_list]

key_embeddings = model.encode(new_keys_list)

top_n = 5
distances = cosine_similarity(doc_embedding, key_embeddings)
keywords = [keys_list[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)