---
layout: post
title: "OpenAI Key 설정"
date: 2024-03-25 16:42:00 +0900
categories: [konkuk, rag]

---
# 🔑 OpenAI API 키 발급 및 설정 방법 요약

---

## ✅ OpenAI 계정 생성 및 로그인

1. [OpenAI API 웹사이트](https://platform.openai.com/docs/overview)에 접속합니다.
2. 우측 상단에서 **`Sign Up`** 을 눌러 회원가입하거나 이미 가입된 경우 **`Log in`**으로 로그인합니다.

---

## 💳 신용카드 등록 및 크레딧 충전 방법

1. 로그인 후, 우측 상단 **`Setting (톱니바퀴)`** 메뉴로 이동합니다.
2. 왼쪽 메뉴의 **`Billing > Payment methods`**를 클릭하여 신용카드를 등록합니다.
3. 신용카드 등록 후 **`Add to credit balance`** 버튼을 클릭하여 사용할 금액을 입력합니다. (최소 충전 금액은 **$5**부터 가능)
4. **`Continue`**를 눌러 결제를 완료합니다.

---

## 📊 월간 사용한도 설정 방법 (Limits 설정)

왼쪽 메뉴의 **`Limits`** 탭에서 설정할 수 있습니다.

- **`Set a monthly budget`**  
  월간 사용한도를 설정하여 설정한 금액에 도달하면 더 이상 과금되지 않고 API 사용이 중지됩니다.

- **`Set an email notification threshold`**  
  지정된 금액에 도달하면 이메일 알림이 발송됩니다.

---

## 🔐 API 키 발급 및 관리 방법

1. 우측 상단 프로필 아이콘에서 **`Your profile`**을 선택하여 API 키 관리 메뉴로 이동합니다.
   - 직접 이동 링크: [API Keys 페이지](https://platform.openai.com/api-keys)

2. **`Create new secret key`**를 클릭하여 키를 생성합니다.
   - 키 생성 시 **Name**과 **프로젝트**를 입력합니다. 별도의 프로젝트가 없다면 **Default project**로 설정합니다.

3. 생성된 키를 우측의 **`Copy`** 버튼으로 복사합니다.

---

## ⚠️ API 키 보안 주의사항

- API 키가 유출되면 타인이 GPT를 사용할 수 있으며, 결제는 본인의 계정에서 이루어집니다.
- 키는 반드시 타인과 공유하지 말고, 안전한 장소에 보관하세요. (비밀번호처럼 관리)

---

## ⚙️ `.env` 파일에 API 키 저장하기

복사한 키를 `.env` 파일에 다음과 같이 저장한 후, 파일을 저장하고 닫습니다.

```bash  
OPENAI_API_KEY=복사한_API_KEY
