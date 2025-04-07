---
layout: post
title: "임베딩 (Embedding)"
date: 2025-04-07 11:52:00 +0900
categories: [konkuk, rag]
--- 

# 임베딩 (Embedding)

**임베딩**은 Retrieval-Augmented Generation(RAG) 시스템의 세 번째 단계로, 문서분할 단계에서 생성된 문서 단위들을 기계가 이해할 수 있는 수치적 형태로 변환하는 과정입니다.

이 단계는 RAG 시스템의 핵심적인 부분 중 하나로, 문서의 의미를 벡터(숫자의 배열) 형태로 표현함으로써, 사용자가 입력한 질문(Query)에 대하여 DB에 저장한 문서 조각(단락, Chunk)을 검색하고 유사도를 계산하는 데 활용됩니다.

---

## 📌 임베딩의 필요성

1. **의미 이해**  
   자연 언어는 매우 복잡하고 다양한 의미를 내포하고 있습니다. 임베딩을 통해 이러한 텍스트를 정량화된 형태로 변환함으로써, 컴퓨터가 문서의 내용과 의미를 더 잘 이해하고 처리할 수 있습니다.

2. **정보 검색 향상**  
   수치화된 벡터 형태로의 변환은 문서 간 유사성을 계산하는 데 필수적입니다. 이는 관련 문서를 검색하거나 질문에 가장 적합한 문서를 찾는 작업을 용이하게 합니다.

---

## 💡 예시
<div style="text-align: center;">
  <img src="/assets/images/mustree/임베딩1.png">
</div>  
<div style="text-align: center;">
  <img src="/assets/images/mustree/임베딩2.png">
</div>

> 질문: 시장조사기관 IDC가 예측한 AI 소프트웨어 시장의 연평균 성장률은 어떻게 되나요?
[0.1, 0.5, 0.9,..., 0.2, 0.4]  

### 📊 임베딩된 문장을 수치 표현으로 변환

| 단락 번호 | 임베딩 벡터 (예시)             |
|-----------|-------------------------------|
| 1번 단락   | 0.1, 0.5, 0.9, ..., 0.1, 0.2   |
| 2번 단락   | 0.7, 0.1, 0.3, ..., 0.5, 0.6   |
| 3번 단락   | 0.9, 0.4, 0.5, ..., 0.4, 0.3   |

### 🔍 유사도 계산 결과

| 단락 번호 | 유사도 점수 | 선택 여부 |
|-----------|-------------|------------|
| 1번 단락   | 80%         | ✅ 선택됨  |
| 2번 단락   | 30%         |            |
| 3번 단락   | 25%         |            |

---

## 💻 코드 예시

```python
# 단계 3: 임베딩(Embedding) 생성
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
```

---

## 📚 참고 자료

- [LangChain Text Embeddings 위키독스](https://wikidocs.net/233815)
- [LangChain 공식 문서 - 텍스트 임베딩](https://python.langchain.com/v0.1/docs/modules/data_connection/text_embedding/)
