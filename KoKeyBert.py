# -*- coding: utf-8 -*-
import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

doc = """
         인천 청라 아파트 단지 지하주차장 전기차 화재가 사람들을 공포에 빠뜨렸다. 이번 인천 전기차 화재 피해는 가공할 만하다. 아파트 지하주차장에 주차한 차 수십대가 전기차 화재로 인해 전소됐다. 이뿐만 아니라 화재로 인해 아파트 단지 전기와 수도까지 상당 기간 공급이 중단돼 주민들이 큰 불편을 겪어야 했다. 전기차 화재 초기 진화에 실패하면서 지하주차장에 매설된 아파트 전기 수도 등 기반시설들도 모두 불탔기 때문이다. 이번 화재의 정확한 원인은 수사기관에서 조사 중이지만 전기차 배터리 안전성 문제가 원인으로 지목된다.
전기차 화재의 위험성과 심각성에 대해 이미 전조증상은 있었다. 이번 화재에 앞서 크고 작은 전기차 화재 사고가 있었고 그때마다 전기차 배터리에서 발생한 불을 끄지 못해 전소하거나 진화에 장시간이 걸리는 문제가 확인됐다. 상당수 전기차 화재가 충전 중에 발생하고 있다는 점 때문에 이에 대한 예방책도 마련돼야 한다는 주장도 제기돼 왔다.
경기일보도 로컬이슈 리포트(2023년 3월17일자 1·3면) 등을 통해 전기차 화재의 위험성과 대책 마련이 시급하다고 여러 차례 경고한 터라 이번 인천 청라 아파트 전기차 화재가 더 안타깝게 다가온다.
경기일보가 주목한 부분은 전기차 충전시설이 아파트 등 지하주차장에 급속도로 설치되고 있다는 점이었다. 안전대책이 없는 상황에서 폐쇄적인 지하주차장에 전기충전시설을 설치할 때 전기차 화재 발생 시 피해는 더 클 수밖에 없다. 그러나 친환경 전기차 활성화 정책을 펴고 있는 정부는 전기차 이용자들의 편의를 위해 아파트 단지 지하 등에 전기차 충전시설 설치를 권장하고 있다.
그러나 확인 결과 안전대책은 역시 미흡했다. 당시 기자들이 아파트 단지 등에 설치된 전기차 충전시설 몇 곳을 돌아봤는데 화재 발생 시 속수무책일 수밖에 없었다. 지하주차장 특성상 화재 발생 시 소방차가 진입할 수 없는 구조이거나 그 흔한 분말소화기조차 비치하지 않은 곳도 있었다. 전기차 배터리 화재는 일반 소화기로는 사실상 진압할 수 없다. 전기차 배터리에서 열폭주를 일으키며 피해는 걷잡을 수 없이 커진다. 전문가들은 전기차 안전 관련 법 제도 강화와 화재 발생 시 신속 진화할 수 있는 시스템 개발 등을 제안했다.
당시 경기일보 기사를 보고 부천시에서 수원 본사까지 찾아 문의하는 독자가 있을 정도로 전기차 화재에 대한 시민들의 공포는 퍼져 있는 상황이었다.
인천 청라 전기차 화재 이후 전기차에 대한 불안감은 더욱 확산되는 양상이다. 전기차 출입을 금지하는 아파트 단지가 등장했다. 정부와 지자체가 뒤늦게 대책 마련에 분주하다. 전기차 생산업체들은 배터리 제조사를 공개하기 시작했다.
인천 청라 전기차 화재 사고 전 전기차 화재의 위험성과 심각성은 모두 알고 있었다. 이에 대한 경고음도 이미 수차례 울린 상황이다. 그러나 우물쭈물하는 사이 대형 사고는 여지없이 발생했다. 언제까지 큰 희생을 치른 뒤 대책이 마련되는 상황을 경험해야 하는지 시민들은 답답하기만 하다.
      """

##    형태소 분석
okt = Okt()
nouns = okt.nouns(doc)

text = ' '.join(nouns)

##    N Gram 생성
n_gram_range = (2,3)

count = CountVectorizer(ngram_range=n_gram_range).fit([text])
candidates = count.get_feature_names_out()

print('trigram 개수 :',len(candidates))
print('trigram 다섯개만 출력 :',candidates[:5])

##    SBERT 사용해 수치화
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)
print(f'doc_embedding : {doc_embedding}')
print(f'\ncandidate_embedding : {candidate_embeddings}')

##    상위 5개 출력
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]

print(keywords)

##    MMR 기법 사용해 다양성 최대화
def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)
    print(f'word_similarity : {word_similarity}')

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]
  
print(mmr(doc_embedding, candidate_embeddings, candidates, top_n=5, diversity=0.5))