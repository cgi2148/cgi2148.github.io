---
layout: post
title: "Deepguard: FacePoison으로 만드는 실전 딥페이크 방어"
date: 2025-09-30 23:59:00 +0900
categories: [newruns]
---

## 왜 FacePoison인가
- 딥페이크 제작 파이프라인은 대부분 얼굴 검출기의 정확도에 의존합니다. 검출 단계에서 얼굴을 놓치면 뒤따르는 합성 과정 자체가 실패합니다.
- FacePoison은 RetinaFace, PyramidBox, S3FD, DSFD, YOLO5Face 다섯 가지 검출기를 동시에 교란하도록 설계된 연구 결과물입니다.
- 노이즈가 육안으로 거의 보이지 않기 때문에, 사용자는 원본 이미지를 유지하면서도 합성 모델의 공격을 실질적으로 차단할 수 있습니다.
- 특히 SNS에 업로드되는 프로필 사진처럼 재사용 빈도가 높은 이미지에 FacePoison을 적용하면, 데이터 수집 시점에서부터 사전 차단을 구현할 수 있습니다.

## 구축 배경과 목표
- **목표**: 딥페이크 대응 기술을 연구 팀 내부에만 머물게 하지 않고, 비전공자도 클릭 몇 번으로 사용할 수 있는 서비스 형태로 제공하기.
- **사용자 스토리**: 팀 내 커뮤니케이션 담당자가 긴급 보도자료 이미지를 업로드하면 몇 초 안에 보호된 버전을 내려받을 수 있어야 한다.
- **성공 지표**: ① 보호 이미지 생성 시간 15초 이하, ② 다중 검출기 교란 성공률 95% 이상, ③ 운영자가 1시간 이내에 환경을 재구축할 수 있는 문서화 수준 확보.

<div style="text-align: center;">
  <img src="/assets/images/newruns/딥가드 메인.png" width="1000" alt="Deepguard 대시보드 메인 화면">
  <p style="margin-top: 10px;"></p>
</div>

## 시스템 아키텍처 한눈에 보기
```
project-root/
├── services/
│   └── deepguard-dashboard/      # Flask 앱
│       ├── app.py                # 엔트리포인트
│       ├── facepoison_processor.py
│       ├── templates/
│       ├── static/
│       ├── uploads/
│       └── results/
├── engines/
│   └── facepoison-core/          # 연구용 FacePoison 원본
│       ├── detectors/
│       ├── attacks/
│       ├── pretrained_weights/
│       └── experiment_scripts/
└── infra/
    └── envs/                     # 가상환경 및 의존성 정의
```
- `services/deepguard-dashboard`는 사용성을 극대화하도록 Flask 기반으로 단순화했고, 비즈니스 로직은 `facepoison_processor.py`에 집중시켰습니다.
- `engines/facepoison-core`는 논문 구현체를 거의 그대로 두면서, 가중치 경로와 추론 옵션을 설정 파일로 분리해 추후 교체가 쉬운 구조로 재구성했습니다.
- 환경 구성은 `infra/envs/facepoison.yml`에 정리하여 Conda, venv 모두에서 재현 가능하도록 맞췄습니다.

## 사용자 흐름 상세 설명
1. **이미지 업로드**  
   사용자가 대시보드에서 이미지를 드래그 앤 드롭하면 `uploads/`에 저장됩니다. 업로드 훅에서 파일명을 UUID로 재정의해 충돌을 방지합니다.
2. **옵션 선택**  
   프리셋으로 `균형형`, `고강도`, `소형 얼굴 최적화`를 제공하고, 세부 옵션(검출기, epsilon 값, 반복 횟수)은 고급 모드에서 노출합니다.
3. **FacePoison 처리**  
   `facepoison_processor.py`가 Celery 워커로 요청을 전달하고, 워커는 GPU 사용 가능 여부에 따라 half precision 모드와 배치 크기를 동적으로 조정합니다.
4. **결과 생성 및 비교**  
   처리된 이미지는 `results/`에 저장되며, 웹소켓으로 프런트엔드에 이벤트를 보내 원본/결과 비교 뷰를 즉시 업데이트합니다.
5. **내려받기 및 정리**  
   사용자에게는 보호 이미지와 JSON 형태의 처리 로그가 제공되며, 24시간이 지나면 주기적으로 저장 공간을 비우는 정리 작업이 실행됩니다.
<div style="text-align: center;">
  <img src="/assets/images/newruns/딥가드_이미지1.png" width="1000" alt="Deepguard 처리 옵션 및 결과 비교 예시">
  <p style="margin-top: 10px;"></p>
</div>

## 실행 체크리스트 (가상 경로 기준)
```bash
# 1. 환경 변수 설정
export PROJECT_ROOT=~/workspace/deepguard
export VENV_PATH=$PROJECT_ROOT/.venv/facepoison

# 2. 가상환경 준비
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install -r "$PROJECT_ROOT/infra/envs/facepoison-requirements.txt"

# 3. 서비스 실행
cd "$PROJECT_ROOT/services/deepguard-dashboard"
export NUMEXPR_MAX_THREADS=8
python app.py
```
- 기본 포트는 5000이며, Reverse Proxy 환경을 고려해 `HOST`, `PORT`, `PREFERRED_URL_SCHEME` 값을 `.env`로 분리했습니다.
- 모델 가중치는 `engines/facepoison-core/pretrained_weights`에, 각 검출기의 개별 가중치는 `engines/facepoison-core/detectors/**/weights`에 배치합니다.
- CUDA가 없는 환경에서는 요구 사양이 낮은 `cpu` 프로필을 활성화해 PyTorch CPU 휠을 설치하도록 안내합니다.

## FacePoison 백엔드 운영 노하우
- **모듈화된 탐지기 레이어**: 새로운 얼굴 검출기를 추가할 때는 `detectors/<name>/weights`와 `detectors/<name>/predictor.py` 패턴만 맞추면 됩니다.
- **실험 파라미터 재현**: `experiment_scripts/configs/*.yaml`에 실험 단위를 정의해두고, 운영 환경에서는 변형된 설정을 그대로 로드합니다.
- **성능 측정**: 변조된 이미지와 원본 이미지 사이의 PSNR, SSIM을 계산해 시각적 품질을 관리하고, 교란 성공 여부는 각 검출기의 detection rate로 기록합니다.
- **대체 경로 지원**: 워커는 가중치를 로컬 캐시에 먼저 찾고, 없으면 사설 객체 스토리지에서 한 번만 내려받도록 구현해 Colab/온프레 환경 모두 대응했습니다.

## UX와 커뮤니케이션 전략
- 업로드 단계에서 예상 처리 시간을 실시간으로 추정해 보여주고, 처리 중에는 검출기별 진행률을 표시해 사용자 불안을 줄였습니다.
- 비교 뷰에는 hover 시 픽셀 단위 차이를 강조하는 heatmap 모드를 추가해 FacePoison이 어떤 영역에 영향을 주었는지 시각적으로 확인할 수 있게 했습니다.
- 결과 카드에는 QR 코드도 함께 표시해 모바일 환경에서 바로 결과를 검증하거나 공유할 수 있도록 했습니다.

## 관측과 자동 복구
- Prometheus + Grafana 대시보드로 GPU/CPU 자원, 처리 시간 분포, 실패율을 추적합니다.
- 워커가 연속 3회 이상 실패하면 자동으로 `–precision half` 옵션을 비활성화하고, 문제가 해결되면 다시 원복합니다.
- 업로드 폴더는 24시간마다 Lambda 크론(또는 crontab)으로 정리하여 저장 공간 폭증을 막고, 동시에 최근 100건의 처리 로그를 별도 저장소에 백업합니다.

## 동영상 보호처리(프레임 기반)
- 처리 파이프라인: 입력 영상 디코딩 → 프레임별 보호 필터(적대적 노이즈/마스크) 적용 → (옵션) 얼굴 영역 선택적 보호 → 원본 오디오 보존 재인코딩.  
- 시간적 일관성: 동일 인물 추적과 프레임 간 스무딩으로 깜빡임을 억제하고, 씬 전환 시 보호 강도를 자동 조정합니다.  
- 호환성: mp4, mov 등 표준 컨테이너를 지원하며, 결과는 동일 해상도·프레임레이트로 반환됩니다.  

<div style="text-align: center;">
  <video controls width="1000" preload="metadata" poster="/assets/images/newruns/딥가드 메인.png">
    <source src="/assets/images/newruns/0922시연.mov">
    브라우저가 동영상을 지원하지 않습니다. <a href="/assets/images/newruns/0922시연.mov">동영상 다운로드</a>
  </video>
  <p style="margin-top: 10px;"></p>
</div>

## 실전 적용 사례: 보도자료 이미지 보호
- 한 언론사의 보도자료가 배포되기 직전, 인물 사진 한 장에 대한 보호 요청이 들어왔습니다.  
  1. 커뮤니케이션 팀이 이미지를 업로드하고 `균형형` 프리셋을 선택  
  2. 프로세스가 9초 만에 완료, RetinaFace와 DSFD 모두에서 detection rate 0% 확인  
  3. 결과 파일을 그대로 보도자료에 첨부해 배포했고, 이후 크롤링된 이미지를 역추적했을 때 딥페이크 합성 시도가 감지되지 않았습니다.
- 이 경험 이후, 반복 작업을 줄이기 위해 REST API 엔드포인트(`POST /api/protect`)를 추가했고, 사내 CMS에서 자동으로 보호 필터를 거칠 수 있게 연동했습니다.

## 트러블슈팅과 배운 점
- PyTorch와 dlib 빌드 충돌로 배포가 지연된 적이 있어, **환경 분리** 원칙을 더욱 엄격하게 지켰습니다. 환경 파일은 `envs/facepoison.lock`으로 고정하고, CI에서 매 릴리스마다 재빌드 테스트를 수행합니다.
- GPU 메모리 OOM 이슈가 발생했을 때는 half precision을 적극 활용하고, 반복 횟수를 동적으로 조절하는 로직을 추가해 안정성을 확보했습니다.
- 문서화는 운영 속도를 좌우했습니다. 설치 체크리스트, 가중치 매핑 표, 일반적인 오류 대응 FAQ를 한 문서에서 찾을 수 있게 하니 신규 운영자가 바로 업무를 이어받을 수 있었습니다.

## 앞으로의 계획
- 결과 이미지에 워터마크와 보호 이력을 심는 **신뢰성 증명 모듈**을 추가해, 합성 공격자가 이미지를 수정했는지 여부를 쉽게 검증하도록 만들 예정입니다.
- FastAPI 기반 REST 버전을 별도 서비스로 분리해, 서버리스나 파이프라인 처리 환경에서도 FacePoison을 적용할 수 있도록 확장합니다.
- 장기적으로는 FacePoison과 유사한 LoRA 기반 경량 모델을 학습시켜 모바일 환경에서도 실시간 보호를 제공하는 것이 목표입니다.
