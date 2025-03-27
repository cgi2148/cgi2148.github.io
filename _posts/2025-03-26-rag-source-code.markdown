---
layout: post
title: "OpenAI API Key 소스코드"
date: 2025-03-26 16:09:00 +0900
categories: [konkuk, rag-code]
---
## OpenAI API 키 발급 및 설정


1. OpenAI API 키 발급

- [OpenAI API 키 발급방법](https://wikidocs.net/233342) 글을 참고해 주세요.

1. `.env` 파일 설정

- 프로젝트 루트 디렉토리에 `.env` 파일을 생성합니다.
- 파일에 API 키를 다음 형식으로 저장합니다:  
  `OPENAI_API_KEY` 에 발급받은 API KEY 를 입력합니다.

- `.env` 파일에 발급한 API KEY 를 입력합니다.

```python
# LangChain 업데이트
# !pip install -r https://raw.githubusercontent.com/teddylee777/langchain-kr/main/requirements.txt
```

```python
# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()
```

API Key 가 잘 설정되었는지 확인합니다.

```python
import os

print(f"[API KEY]\n{os.environ['OPENAI_API_KEY'][:-15]}" + "*" * 15)
```

설치된 LangChain 버전을 확인합니다.

```python
from importlib.metadata import version

print("[LangChain 관련 패키지 버전]")
for package_name in [
    "langchain",
    "langchain-core",
    "langchain-experimental",
    "langchain-community",
    "langchain-openai",
    "langchain-teddynote",
    "langchain-huggingface",
    "langchain-google-genai",
    "langchain-anthropic",
    "langchain-cohere",
    "langchain-chroma",
    "langchain-elasticsearch",
    "langchain-upstage",
    "langchain-cohere",
    "langchain-milvus",
    "langchain-text-splitters",
]:
    try:
        package_version = version(package_name)ㄱ
        print(f"{package_name}: {package_version}")
    except ImportError:
        print(f"{package_name}: 설치되지 않음")
```