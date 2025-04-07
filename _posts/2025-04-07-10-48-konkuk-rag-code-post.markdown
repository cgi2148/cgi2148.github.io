---
layout: post
title: "텍스트 분할 (Text Splitter)"
date: 2025-04-07 10:48:00 +0900
categories: [konkuk, rag]
--- 
# 텍스트 분할 (Text Splitter)

**문서 분할**은 Retrieval-Augmented Generation(RAG) 시스템의 두 번째 단계로서, 로드된 문서들을 효율적으로 처리하고 시스템이 정보를 보다 잘 활용할 수 있도록 준비하는 중요한 과정입니다.

이 단계의 목적은 크고 복잡한 문서를 LLM이 받아들일 수 있는 효율적인 작은 규모의 조각으로 나누는 작업입니다.  
→ 나중에 사용자가 입력한 질문에 대해 보다 **효율적인 정보만 압축·선별하여 가져오기 위함**입니다.

---

## 📌 예시 질문

> 구글이 앤스로픽에 투자한 금액은 얼마야?  
<div style="text-align: center;">
  <img src="/assets/images/mustree/텍스트분할 1.png">
</div>
---

## 문서 분할의 필요성

1. **핀포인트 정보 검색 정확성**  
   → 문서를 세분화함으로써 질문(Query)에 연관성이 있는 정보만 가져오는데 도움을 줍니다.  
   → 각각의 단위는 특정 주제나 내용에 초점을 맞추므로 관련성이 높은 정보를 제공합니다.

2. **리소스 최적화 효율성**  
   → 전체 문서를 LLM으로 입력하게 되면 비용이 많이 발생할 뿐더러,  
   → 효율적인 답변을 많은 정보 속에서 발췌하지 못하게 됩니다.  
   → 때로는 이러한 문제가 **할루시네이션(hallucination)**으로 이어지기도 합니다.

---

## 🔧 문서 분할 과정

1. **문서 구조 파악**  
   → PDF, 웹 페이지, 전자책 등 다양한 형식의 문서에서 구조를 파악  
   → 예: 헤더, 푸터, 페이지 번호, 섹션 제목 등 식별

2. **단위 선정**  
   → 문서를 어떤 단위로 나눌지 결정 (페이지별, 섹션별, 문단별 등)

3. **청크 크기 선정 (chunk size)**  
   → 문서를 몇 개의 토큰 단위로 나눌 것인지 결정

4. **청크 오버랩 (chunk overlap)**  
   → 분할된 끝부분에서 맥락이 이어질 수 있도록 일부를 겹쳐서 분할하는 것이 일반적

---

## 🔁 청크 크기 & 청크 오버랩  
<div style="text-align: center;">
  <img src="/assets/images/mustree/텍스트분할 2.png">
</div>  

| 설정 항목         | 설명                                                    |
|------------------|---------------------------------------------------------|
| `chunk_size`     | 한 조각(청크)의 최대 토큰 수                           |
| `chunk_overlap`  | 이전 청크와 겹치는 토큰 수 (문맥 연결 유지용)          |

---

## 💻 코드 예시

```python
# 단계 2: 문서 분할 (Split Documents)
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

splits = text_splitter.split_documents(docs)
