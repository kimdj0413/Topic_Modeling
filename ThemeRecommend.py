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
창고에서 화재가 발생하자 배터리를 옮기던 회사 대표가 폭발로 인해 사망했다.

22일 오전 8시 35분께 충북 진천군 진천읍의 한 산업용 배터리 도매 창고에서 화재가 발생했다.

이에 업체 대표 A(40대)씨가 지게차로 배터리를 바깥으로 옮기려다 폭발로 인해 쓰러진 것으로 전해졌다.

배터리는 스쿠터용 이차전지 리튬 배터리로 추정됐다.

소방당국은 인력 41명과 장비 24대를 투입해 2시간 30여분 만에 불을 진화했다.

배터리 도매 창고 1개 동(286.7㎡)이 전소되면서 안에 있던 배터리 2천개가 불에 탔다. 바로 옆에 있던 플라스틱 필름 창고(988.8㎡) 일부도 그을렸다.

소방당국은 A씨가 지게차로 배터리를 옮기던 중 배터리를 바닥에 떨어뜨려 불이 시작된 것으로 보고 정확한 화재 원인과 피해 규모를 조사 중이다.

소방 관계자는 "리튬 전지는 액체 전해질이 분리막에 의해 음극과 양극으로 나뉘어 있는 구조인데, 외부 충격으로 분리막이 훼손되면 액체 전해질이 흐르면서 불이 나거나 폭발할 위험이 높다"면서 "일단 불이 나면 열폭주 현상을 일으키기 때문에 신속히 대피한 뒤 119에 신고해야 한다"고 설명했다.
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