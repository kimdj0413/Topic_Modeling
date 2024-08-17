import re

doc = """
         올해 상반기 매출액은 1조2287억원, 영업이익은 598억원으로 전년동기대비 각각 12.31%, 48.76% 불어났다. 

반도체 경기 악화로 지난해 대폭 감소했던 배당수익이 올해 예년 수준으로 회복된 데 따른 기저효과로 풀이된다. 이를 제외하더라도 생성형 인공지능(AI), 클라우드 등 IT 서비스 사업에서 안정적인 성장세를 보였다.

SK C&C는 올해 1분기에 이어 2분기에도 산업별 대규모 디지털전환(DX) 시스템 구축을 비롯해 생성형 AI, 클라우드, 디지털 팩토리, 디지털 ESG 등 디지털 ITS 사업 전반에 걸쳐 성과를 내고 있다.

현대홈쇼핑 차세대 시스템 구현을 시작으로 CJ대한통운의 새로운 택배 시스템 '로이스 파슬'에 클라우드 네이티브를 기반으로 하는 디지털 택배 체계를 구축하고 엔터프라이즈 AI솔루션인 '솔루어(Solur)'를 활용한 하이퍼오토메이션 구현에도 적극 나섰다.

최근에는 AI개인화 서비스 '마이박스(Mybox)'를 내놓으며, 기업 구성원들이 자신의 업무에 맞춰 AI 챗봇을 수시로 생성·활용할 수 있도록 하고 있다. 

SK C&C 관계자는 "'엔터프라이즈 AI 서비스'를 통해 기업에 맞는 최적의 AI 레시피를 제공하며 기업의 AI 네이티브 혁신과 비즈니스 가치 창출을 이끌겠다"고 전했다.  
      """

split_text = re.split(r'(\s*[a-zA-Z]+(?:[^\w\s]+[a-zA-Z]+)*\s*)', doc)

# 불필요한 공백 제거
split_text = [part.strip() for part in split_text if part.strip()]

# 결과 출력
for i, part in enumerate(split_text):
    print(f"부분 {i + 1}: '{part}'")