---
layout: post
title: "컨텍스트 윈도우(Context Window)와 출력 토큰"
date: 2024-03-25 17:02:00 +0900
categories: [konkuk, rag]
---
# 컨텍스트 윈도우(Context Window)와 출력 토큰 🔍

## 컨텍스트 윈도우(Context Window)

- **컨텍스트 윈도우**란, 모델이 한 번에 처리할 수 있는 최대 토큰(입력+출력)의 길이를 의미합니다.
- **Context Length**라고도 하며, 긴 텍스트를 처리하는 데 중요한 개념입니다.
- GPT 모델별 컨텍스트 길이:
  - **GPT-3.5**: 약 16K 토큰
  - **GPT-4**:
    - 기본 버전: 8,192 토큰
    - 확장 버전: 최대 32,768 토큰

## 최대 출력 토큰(max_tokens)

- 모델이 답변으로 생성할 수 있는 최대 출력 토큰 수를 의미하는 매개변수입니다.
- 예시: `max_tokens`를 100으로 설정하면, 모델은 최대 100개의 토큰까지 출력합니다.
- 주의: 이 값은 모델의 컨텍스트 길이(Context Length)를 초과할 수 없습니다.

## 토큰의 길이와 비용 💰

- 4,096 토큰은 한글 약 1,350자(단어가 아닌 글자 수 기준)이며, 이는 워드 문서 약 3장 정도의 분량에 해당합니다.
- 입력 토큰과 출력 토큰은 비용이 다르게 측정됩니다. 정확한 비용 산정을 위해서는 구조에 대한 이해가 중요하며, 불필요한 비용 증가를 방지할 수 있습니다.

## 유용한 도구 및 참고 자료 📌

- [OpenAI 모델 공식 문서](https://platform.openai.com/docs/models)  
- [GPT-4o 비용 계산기](https://livechatai.com/gpt-4o-pricing-calculator)  
- [OpenAI 비용 계산기](https://invertedstone.com/calculators/openai-pricing/)  
  
모델을 효율적으로 활용하기 위해 컨텍스트 윈도우와 최대 출력 토큰 설정에 대한 충분한 이해가 필요합니다. 🚀

