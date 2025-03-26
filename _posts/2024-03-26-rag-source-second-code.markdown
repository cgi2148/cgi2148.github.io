---
layout: post
title: "OpenAI-LLM 소스코드"
date: 2024-03-26 17:09:00 +0900
categories: [konkuk, rag-code]
---
네, Jupyter Notebook의 내용을 정확하게 마크다운으로 변환하겠습니다. 파일명에 공백이 있어서 먼저 수정하고, 내용을 순차적으로 변환하겠습니다.

먼저 파일명을 수정하겠습니다:


```bash
mv "_posts/2024-03-26-rag-source-second-code .markdown" "_posts/2024-03-26-rag-source-second-code.markdown"
```
이제 파일의 내용을 정확하게 수정하겠습니다:


```markdown:_posts/2024-03-26-rag-source-second-code.markdown
---
layout: post
title: "OpenAI-LLM 소스코드"
date: 2024-03-26 17:09:00 +0900
categories: [konkuk, rag-code]
---

# OpenAI-LLM 소스코드

이 포스트에서는 OpenAI의 LLM(Large Language Model)을 사용하는 방법에 대해 설명합니다.

## ChatOpenAI

OpenAI 사의 채팅 전용 Large Language Model(llm) 입니다.

객체를 생성할 때 다음을 옵션 값을 지정할 수 있습니다. 옵션에 대한 상세 설명은 다음과 같습니다.

### 주요 옵션

#### temperature
- 사용할 샘플링 온도는 0과 2 사이에서 선택합니다.
- 0.8과 같은 높은 값은 출력을 더 무작위하게 만들고
- 0.2와 같은 낮은 값은 출력을 더 집중되고 결정론적으로 만듭니다.

#### max_tokens
- 채팅 완성에서 생성할 토큰의 최대 개수입니다.

#### model_name
적용 가능한 모델 리스트:
- `gpt-4o`
- `gpt-4o-mini`
- `o1-preview`, `o1-preview-mini`: tier5 계정 이상만 사용 가능

![gpt-models.png](./images/gpt-models2.png)

- 링크: https://platform.openai.com/docs/models

## 기본 사용법

```python
from langchain_openai import ChatOpenAI

# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    model_name="gpt-4o-mini",  # 모델명
)

# 질의내용
question = "대한민국의 수도는 어디인가요?"

# 질의
print(f"[답변]: {llm.invoke(question)}")
```

## 응답 형식

LLM의 응답은 다음과 같은 형식으로 반환됩니다:

```python
# 응답 객체의 구조
response = llm.invoke(question)
print(response.content)  # 실제 응답 내용
print(response.response_metadata)  # 메타데이터 정보
```

## LogProb 활성화

주어진 텍스트에 대한 모델의 **토큰 확률의 로그 값**을 의미합니다. 토큰이란 문장을 구성하는 개별 단어나 문자 등의 요소를 의미하고, 확률은 **모델이 그 토큰을 예측할 확률**을 나타냅니다.

```python
# LogProb 활성화된 객체 생성
llm_with_logprob = ChatOpenAI(
    temperature=0.1,
    max_tokens=2048,
    model_name="gpt-4o-mini",
).bind(logprobs=True)

# 질의 및 결과 확인
response = llm_with_logprob.invoke(question)
print(response.response_metadata)
```

## 스트리밍 출력

스트리밍 옵션은 질의에 대한 답변을 실시간으로 받을 때 유용합니다.

```python
# 스트림 방식으로 질의
answer = llm.stream("대한민국의 아름다운 관광지 10곳과 주소를 알려주세요!")

# 스트리밍 방식으로 각 토큰을 출력
for token in answer:
    print(token.content, end="", flush=True)
```

```python
from langchain_teddynote.messages import stream_response

# 스트림 방식으로 질의
answer = llm.stream("대한민국의 아름다운 관광지 10곳과 주소를 알려주세요!")
stream_response(answer)
```

## 프롬프트 캐싱

프롬프트 캐싱 기능을 활용하면 반복하여 동일하게 입력으로 들어가는 토큰에 대한 비용을 아낄 수 있습니다.

다만, 캐싱에 활용할 토큰은 고정된 PREFIX를 주는 것이 권장됩니다.

```python
from langchain_teddynote.messages import stream_response

very_long_prompt = """
당신은 매우 친절한 AI 어시스턴트 입니다. 
당신의 임무는 주어진 질문에 대해 친절하게 답변하는 것입니다.

<WANT_TO_CACHE_HERE>
#참고:
**Prompt Caching**
Model prompts often contain repetitive content, like system prompts and common instructions. OpenAI routes API requests to servers that recently processed the same prompt, making it cheaper and faster than processing a prompt from scratch. This can reduce latency by up to 80% and cost by 50% for long prompts. Prompt Caching works automatically on all your API requests (no code changes required) and has no additional fees associated with it.

Prompt Caching is enabled for the following models:

gpt-4o (excludes gpt-4o-2024-05-13 and chatgpt-4o-latest)
gpt-4o-mini
o1-preview
o1-mini
This guide describes how prompt caching works in detail, so that you can optimize your prompts for lower latency and cost.

Structuring prompts
Cache hits are only possible for exact prefix matches within a prompt. To realize caching benefits, place static content like instructions and examples at the beginning of your prompt, and put variable content, such as user-specific information, at the end. This also applies to images and tools, which must be identical between requests.

How it works
Caching is enabled automatically for prompts that are 1024 tokens or longer. When you make an API request, the following steps occur:

Cache Lookup: The system checks if the initial portion (prefix) of your prompt is stored in the cache.
Cache Hit: If a matching prefix is found, the system uses the cached result. This significantly decreases latency and reduces costs.
Cache Miss: If no matching prefix is found, the system processes your full prompt. After processing, the prefix of your prompt is cached for future requests.
Cached prefixes generally remain active for 5 to 10 minutes of inactivity. However, during off-peak periods, caches may persist for up to one hour.

Requirements
Caching is available for prompts containing 1024 tokens or more, with cache hits occurring in increments of 128 tokens. Therefore, the number of cached tokens in a request will always fall within the following sequence: 1024, 1152, 1280, 1408, and so on, depending on the prompt's length.

All requests, including those with fewer than 1024 tokens, will display a cached_tokens field of the usage.prompt_tokens_details chat completions object indicating how many of the prompt tokens were a cache hit. For requests under 1024 tokens, cached_tokens will be zero.

What can be cached
Messages: The complete messages array, encompassing system, user, and assistant interactions.
Images: Images included in user messages, either as links or as base64-encoded data, as well as multiple images can be sent. Ensure the detail parameter is set identically, as it impacts image tokenization.
Tool use: Both the messages array and the list of available tools can be cached, contributing to the minimum 1024 token requirement.
Structured outputs: The structured output schema serves as a prefix to the system message and can be cached.
Best practices
Structure prompts with static or repeated content at the beginning and dynamic content at the end.
Monitor metrics such as cache hit rates, latency, and the percentage of tokens cached to optimize your prompt and caching strategy.
To increase cache hits, use longer prompts and make API requests during off-peak hours, as cache evictions are more frequent during peak times.
Prompts that haven't been used recently are automatically removed from the cache. To minimize evictions, maintain a consistent stream of requests with the same prompt prefix.
Frequently asked questions
How is data privacy maintained for caches?

Prompt caches are not shared between organizations. Only members of the same organization can access caches of identical prompts.

Does Prompt Caching affect output token generation or the final response of the API?

Prompt Caching does not influence the generation of output tokens or the final response provided by the API. Regardless of whether caching is used, the output generated will be identical. This is because only the prompt itself is cached, while the actual response is computed anew each time based on the cached prompt. 

Is there a way to manually clear the cache?

Manual cache clearing is not currently available. Prompts that have not been encountered recently are automatically cleared from the cache. Typical cache evictions occur after 5-10 minutes of inactivity, though sometimes lasting up to a maximum of one hour during off-peak periods.

Will I be expected to pay extra for writing to Prompt Caching?

No. Caching happens automatically, with no explicit action needed or extra cost paid to use the caching feature.

Do cached prompts contribute to TPM rate limits?

Yes, as caching does not affect rate limits.

Is discounting for Prompt Caching available on Scale Tier and the Batch API?

Discounting for Prompt Caching is not available on the Batch API but is available on Scale Tier. With Scale Tier, any tokens that are spilled over to the shared API will also be eligible for caching.

Does Prompt Caching work on Zero Data Retention requests?

Yes, Prompt Caching is compliant with existing Zero Data Retention policies.
</WANT_TO_CACHE_HERE>

#Question:
{}
"""

# 토큰 사용량 확인
with get_openai_callback() as cb:
    answer = llm.invoke(
        very_long_prompt.format("프롬프트 캐싱 기능에 대해 2문장으로 설명하세요")
    )
    print(cb)
    cached_tokens = answer.response_metadata["token_usage"]["prompt_tokens_details"]["cached_tokens"]
    print(f"캐싱된 토큰: {cached_tokens}")
```

## 멀티모달 모델(이미지 인식)

멀티모달은 여러 가지 형태의 정보(모달)를 통합하여 처리하는 기술입니다. 다음과 같은 데이터 유형을 포함할 수 있습니다:

- 텍스트: 문서, 책, 웹 페이지 등의 글자로 된 정보
- 이미지: 사진, 그래픽, 그림 등 시각적 정보
- 오디오: 음성, 음악, 소리 효과 등의 청각적 정보
- 비디오: 동영상 클립, 실시간 스트리밍 등

`gpt-4o` 나 `gpt-4-turbo` 모델은 이미지 인식 기능(Vision)이 추가되어 있는 모델입니다.

```python
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response

# 객체 생성
llm = ChatOpenAI(
    temperature=0.1,
    model_name="gpt-4o",
)

# 멀티모달 객체 생성
multimodal_llm = MultiModal(llm)

# 이미지 URL로부터 질의
IMAGE_URL = "https://t3.ftcdn.net/jpg/03/77/33/96/360_F_377339633_Rtv9I77sSmSNcev8bEcnVxTHrXB4nRJ5.jpg"
answer = multimodal_llm.stream(IMAGE_URL)
stream_response(answer)

# 로컬 이미지 파일로부터 질의
IMAGE_PATH_FROM_FILE = "./images/sample-image.png"
answer = multimodal_llm.stream(IMAGE_PATH_FROM_FILE)
stream_response(answer)
```

## System, User 프롬프트 수정

```python
# 시스템 프롬프트 설정
system_prompt = """당신은 표(재무제표)를 해석하는 금융 AI 어시스턴트 입니다. 
당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""

user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

# 프롬프트가 설정된 멀티모달 객체 생성
multimodal_llm_with_prompt = MultiModal(
    llm, system_prompt=system_prompt, user_prompt=user_prompt
)

# 로컬 PC에 저장되어 있는 이미지의 경로 입력  
IMAGE_PATH_FROM_FILE = "https://storage.googleapis.com/static.fastcampus.co.kr/prod/uploads/202212/080345-661/kwon-01.png"  
  
# 이미지 파일로부터 질의 (스트림 방식)  
answer = multimodal_llm_with_prompt.stream(IMAGE_PATH_FROM_FILE)  
  
# 스트리밍 방식으로 각 토큰을 출력합니다.(실시간 출력)
stream_response(answer)
```

