---
layout: post
title: "판매용 텍스트 자동 생성 시스템 구축기"
date: 2024-09-02 10:00:00 +0900
category: mustree
---

# 판매용 텍스트 자동 생성 시스템 구축기

<div align="center">
![웹 기술 스택]({{ site.baseurl }}/assets/images/mustree/그림1.png)

*HTML, CSS, JavaScript를 기반으로 한 웹 기술 스택*
</div>

## 개요

의류 전자상거래에서 고객에게 매력적인 상품 소개는 구매 전환율을 높이는 핵심 요소입니다. 그러나 수천 개에 이르는 상품마다 별도의 판매 텍스트를 작성하는 것은 막대한 인력과 시간이 소요됩니다.  
이에 본 프로젝트는 생성형 AI를 활용하여 상품 상세페이지의 텍스트를 자동으로 생성하는 시스템을 개발하고, 이를 실제 업무 환경에 적용하여 효율성을 검증했습니다.

Claude 3.5 Sonnet, GPT-4o 등 최신 LLM을 통합하여 사용자 맞춤형 텍스트 자동 생성, UI/UX 구현, 시스템 성능 평가, 개선 피드백 루프 구성까지 전반적인 시스템 구축 및 실험을 수행했습니다.

## 프로젝트 목표

- 생성형 AI 기반 판매 텍스트 자동 생성 시스템 아키텍처 설계 및 구현
- 자동 생성 텍스트와 기존 수기 작성 방식 간 정량적/정성적 성능 비교
- AI와 인간의 협업을 전제로 한 효율적인 생산성 향상 전략 제안
- 텍스트 품질을 평가할 수 있는 구체적인 평가 지표 개발 및 실험

## 주요 기술 및 도구

| 기술/도구            | 활용 내용 |
|---------------------|-----------|
| Claude 3.5 Sonnet   | UI 설계 초안 생성, 설계 조언 및 코드 작성 |
| GPT-4o-mini         | 텍스트 생성의 메인 엔진으로 활용 |
| HTML/CSS/JavaScript | 프론트엔드 인터페이스 개발 |
| OpenAI Assistant API| 텍스트 생성 로직 구성 및 호출 |
| Prompt Engineering  | 키워드 중심, SEO 최적화 프롬프트 구조 설계 |

<div align="center">
![시스템 아키텍처]({{ site.baseurl }}/assets/images/mustree/통신흐름도.png)

*클라이언트-서버-AI API 연동 아키텍처*
</div>

## 구현 과정

### 1. 시스템 아키텍처 설계

- 프론트엔드: 입력값(상품명, 성별, 연령대, 말투 등)을 받아 텍스트를 시각적으로 확인할 수 있는 UI 제공
- 백엔드: Node.js 기반 서버가 OpenAI API와 통신하며 생성된 결과를 반환
- AI 모델: 프롬프트에 사용자 입력을 주입하여 다양한 조건의 맞춤형 문장 생성

### 2. 사용자 인터페이스 개발

<div align="center">
![판매글 생성기 UI - 모던 카드 스타일]({{ site.baseurl }}/assets/images/mustree/모던카드.png)

*판매글 생성기 UI - 모던 카드 스타일 디자인*
</div>

<div align="center">
![판매글 생성기 UI - 입력 폼]({{ site.baseurl }}/assets/images/mustree/무제.png)

*판매글 생성기 UI - 상세 입력 폼*
</div>

- 초기 UI는 Claude를 활용해 마크업 구현
- 사용자의 피드백을 반영하여 필수 입력 항목을 확장 (예: 말투, 계절감 등)

### 3. 프롬프트 설계에서 고려한 핵심 요소

- **SEO 최적화**: 검색 노출을 위한 핵심 키워드 삽입
- **제품 특성 강조**: 의류의 핏, 소재, 계절감 등을 문장 내에 자연스럽게 반영
- **문장 어조 설정**: 타겟 고객층에 맞는 말투 및 분위기 설정  
  예: `캐주얼한 느낌의` vs. `세련된 감성의`

### Claude와 GPT-4o 결과물 비교

| 항목              | Claude                                  | GPT-4o                             |
|-------------------|------------------------------------------|------------------------------------|
| 문장 다양성       | ✅ 표현이 풍부하고 묘사적임                | ◻️ 비교적 간결하고 구조화됨         |
| 불필요한 문장 포함 | ⚠️ 일부 문장 장황(중복 또는 과한 묘사 포함) | ✅ 간결한 표현 유지                 |
| 문법 정확도       | ◻️ 다소 부자연스러운 종결 어미 포함 가능성 | ✅ 한국어 문법 및 종결어미 처리 상대적 우수 |

```plaintext
Prompt List
1.You are an E-commerce SEO expert and a highly ex  perienced marketer and clothin g retailer with extensive experience in selling apparel both in online stores and second-hand markets.
2.our role is to generate keyword-rich, informative, and captivating product summaries that effectively market products to potential buyers. 
3. You write in Korean, using emotional and persuasive language to highlight the product's benefits and appeal. 
4. Your goal is to create engaging content sections with eye-catching subheadings, H1 headings, broad match keywords, and a persuasive Description. 
5. You will receive a brief description of the product, its uses, target age group, desired tone for the writing, product type, style, material, usage scenarios, and any additional notes. 
6. Based on this information, you will adjust the length of the description accordingly and generate a slightly different version with each refresh.
7. If the product is second-hand, include an imaginative backstory detailing why it is being sold and how it was used.
8. You also create concise titles based on the provided information, ensuring they are within 10 to 20 characters, and generate slightly different versions with each refresh.
9. Generate content that reflects the input image.
10. Automatically generate titles and detailed descriptions in a similar format based on the entered knowledge.
11. Focus on emphasizing details and features without the need for complete sentences.
12. Improve readability by appropriately pressing the enter key at relevant sections.
13. Titles should reflect facts rather than subjective evaluations of the product.
14. List only nouns that represent the features of the product in the title.
15. Avoid using words in the title that might negatively affect potential buyers, such as 'beware~'.
16. Unify the tone and final endings of the description (e.g., if using sentence form, use it throughout).
17. Minimize repetitions of identical or similar words or content.
18. Vary the tone based on selected conditions. For example, if 'concise' is chosen, avoid endings like ~입니다, ~합니다, and aim for a tone that ends with nouns.
```

### 4. 서버 구축 및 통신 안정화

- 비동기 처리 및 예외 핸들링 로직 구현
- 입력 부족 시 fallback 프롬프트 사용

### 5. UI 개선 및 테스트

- 다양한 조건별 결과를 비교 가능한 미리보기 기능 추가
- 반응형 레이아웃 및 크로스 브라우징 호환성 확보

## 성능 평가

100개 샘플을 기준으로 다음 항목의 반영 여부를 정량 평가하였으며, 평균 반영률은 약 96.4%로 측정되었습니다.

| 평가 항목      | 반영률       |
|----------------|-------------|
| 상품 요약 설명 | 100%         |
| 성별 표현      | 100%         |
| 연령대 반영    | 100%         |
| 말투 설정      | 92%          |
| 소재 정보      | 100%         |
| 스타일 요소    | 100%         |
| 계절감 표현    | 100%         |

정성적 평가는 문장 자연스러움, 표현의 다양성, 일관성 등을 중심으로 진행하였고, GPT-4o는 문법적 정확성 면에서 우수한 성과를 보였습니다.

## 한계점 및 개선 방향

### 한계점

- 문장 어조 설정이 완전하게 반영되지 않는 경우 발생
- 입력 정보가 구체적이지 않으면 일반적인 결과가 생성됨

### 개선 방안

1. 비전 모델 연동을 통해 이미지 기반 상품 특징 추출
2. 상품 태그 정보(소재, 세탁법 등)를 자동 수집하여 텍스트에 반영
3. 자연스러움, 구매 유도력 등 정성 지표 평가 체계 마련
4. 실제 온라인몰에 적용해 클릭률 및 전환율 기반 A/B 테스트 진행

## 결론 및 향후 방향

본 프로젝트는 생성형 AI가 반복적인 콘텐츠 생산을 자동화할 수 있는 실질적 가능성을 입증했습니다.  
이는 전자상거래 업계에서의 적용뿐 아니라, 다양한 산업 분야에서도 유사한 형태로 확장될 수 있을 것입니다.  
향후에는 이미지 기반 설명 생성, 멀티모달 입력을 활용한 텍스트 품질 향상 등을 중심으로 후속 연구를 진행할 예정입니다.  

## 사용된 기술 스택

- Claude 3.5 Sonnet
- GPT-4o / OpenAI API
- HTML, CSS, JavaScript
- Node.js, Express

## 참고문헌

- Akiyo, N., et al. (2023). "Language Model Applications in Retail Automation"
- Zhang, X., et al. (2022). "Generative Text for E-commerce"
- Dellermann, D., et al. (2019). "Human-AI Collaboration in Business"
- Noy, S., Zhang, W. (2023). "Experimental Evidence on GPT-4's Writing Productivity"

> ✍️ 이 포스트는 실험을 기반으로 한 실무 경험을 바탕으로 작성되었으며,  
생성형 AI 기반 금융 분석에 관심 있는 분들에게 참고가 되기를 바랍니다. 