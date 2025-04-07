---
layout: post
title: "벡터스토어 저장 (Vector Store)"
date: 2025-04-07 12:05:00 +0900
categories: [konkuk, rag]
--- 

# 벡터스토어 저장 (Vector Store)

**벡터스토어 저장** 단계는 Retrieval-Augmented Generation(RAG) 시스템의 네 번째 단계로,  
이전 단계에서 생성된 임베딩 벡터들을 효율적으로 저장하고 관리하는 과정입니다.  
이 단계는 향후 검색 과정에서 벡터들을 빠르게 조회하고, 관련 문서를 신속하게 찾아내는 데 필수적입니다.

---

## 📌 벡터스토어 저장의 필요성

1. **빠른 검색 속도**  
   → 임베딩 벡터들을 효과적으로 저장하고 색인화함으로써, 대량의 데이터 중에서도 관련된 정보를 빠르게 검색할 수 있습니다.

2. **스케일러빌리티(확장성)**  
   → 데이터가 지속적으로 증가함에 따라, 벡터스토어는 이를 수용할 수 있어야 합니다.  
   → 효율적인 저장 구조는 데이터베이스의 확장성을 보장하며, 성능 저하 없이 대규모 데이터를 관리할 수 있게 합니다.

3. **의미 기반 검색(Semantic Search) 지원**  
   → 키워드 기반 검색 대신, 의미상으로 유사한 단락을 검색하는 기능을 제공합니다.  
   → 기존 텍스트 DB는 키워드에 의존하는 반면, 벡터스토어는 의미 유사성 기반으로 검색이 가능합니다.

---

## 💡 예시 질문

> 모바일 디바이스 상에서 동작하는 인공지능 기술을 소개한 기업명은?

---

## 📊 벡터스토어의 중요성

벡터스토어 저장 단계는 RAG 시스템의 **검색 기능**과 직접적으로 연결되어 있으며,  
전체 시스템의 **응답 시간**과 **정확성**에 큰 영향을 미칩니다.  

이 단계를 통해 데이터가 잘 관리되고, 필요할 때 즉시 접근할 수 있어  
사용자에게 신속하고 정확한 정보를 제공할 수 있습니다.

---

## 💻 코드 예시

```python
# 단계 4: DB 생성(Create DB) 및 저장
from langchain_community.vectorstores import FAISS

# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
```

---

## 📚 참고 자료

- [위키독스: 벡터스토어 사용 방법](https://wikidocs.net/234013)
- [LangChain 공식 문서 - Vector Stores](https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/)