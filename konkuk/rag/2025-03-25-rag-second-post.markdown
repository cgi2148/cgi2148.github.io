---
layout: post
title: "LangSmith 설정"
date: 2025-03-25 16:35:00 +0900
categories: [konkuk, rag]
---
# 📌 LangSmith 설정 방법 요약

---

## 🔖 LangSmith란?
**LangSmith**는 LLM 애플리케이션의 개발, 모니터링, 테스트를 지원하는 플랫폼으로, 강력한 **추적(tracing)** 기능을 제공합니다.

LangChain을 사용하든 아니든 상관없이 프로젝트 초기 단계부터 반드시 설정하는 것이 권장됩니다.

---

## 🛠️ LangSmith 추적 기능의 중요성

LangSmith의 추적 기능은 다음과 같은 문제를 진단하고 해결하는 데 도움이 됩니다.

- **프로젝트 단위 추적 가능**
  - 실행 횟수, 에러 발생률, 토큰 사용량, 과금 정보 확인 가능
- **예상치 못한 결과 분석 가능**
  - 에이전트가 루핑되는 이유 분석
  - 체인의 속도 지연 원인 분석
  - 단계별 토큰 사용량 분석 가능

---

## 🔍 추적 결과 예시 (1회 실행 기준)
- 문서의 검색 결과 및 GPT의 입출력 내용을 세부적으로 기록합니다.
- 실행 시간(~30초), 사용 토큰 수(예: 5,104 토큰), 비용까지 직관적으로 표시됩니다.

---

## 🔑 LangSmith API Key 발급 방법

1. [LangSmith 사이트](https://smith.langchain.com)에 가입 및 이메일 인증
2. **`Settings > Personal > Create API Key`**를 눌러 API 키 발급
   - 발급받은 API 키는 별도의 안전한 곳에 보관

---

## ⚙️ `.env` 파일에 API 키 설정하기

발급받은 API 키와 프로젝트 정보를 `.env` 파일에 입력합니다.

```bash
LANGCHAIN_TRACING_V2=true  
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  
LANGCHAIN_API_KEY=발급받은_API_KEY  
LANGCHAIN_PROJECT=프로젝트_이름  
