---
layout: post
title: "data_collection_cli.py: YOLOv8 기반 얼굴 바운딩·키포인트 파이프라인"
date: 2025-09-30 09:30:00 +0900
categories: [newruns]
tags:
---

## 만들게 된 배경
- 150초 단위로 정제된 영상에서 얼굴 정보를 추출할 때 **품질**과 **재현성**은 모두 중요했습니다.  
- 단순히 JSON만 남겨두면 중복 처리 여부를 파악하기 어렵고, 수천 개의 파일에서 오류를 추적하기가 거의 불가능했습니다.  
- GPU가 없는 환경에서도 기본 기능이 동작해야 했기 때문에, YOLO가 실패할 경우 OpenCV 기반 폴백을 마련하는 것이 필수였습니다.

## 구성 요소와 가상 경로
```
~/workspace/deepguard/pipelines/data-collection
├── data_collection_cli.py
├── detectors/                 # YOLOv8-face, OpenCV 등
├── db/face_detection.db       # SQLite 결과 (자동 생성)
├── schemas/                   # DB, JSON 스키마 정의
└── docs/
```
- 입력 영상은 `~/workspace/deepguard/storage/processed-clips`에서 가져오고, JSON 결과는 `~/workspace/deepguard/storage/annotations`에 저장합니다.
- DB는 기본적으로 `pipelines/data-collection/db/face_detection.db`에 생성되지만, CLI 옵션으로 다른 경로를 전달할 수 있습니다.

## CLI 사용법 (예시)
```bash
cd ~/workspace/deepguard/pipelines/data-collection
python data_collection_cli.py \
  --input-dir ~/workspace/deepguard/storage/processed-clips \
  --json-dir ~/workspace/deepguard/storage/annotations \
  --db-path  ~/workspace/deepguard/pipelines/data-collection/db/face_detection.db \
  --device auto --conf 0.3 --max-dim 1280
```
- **세션 분할**: `--session-id 0 --total-sessions 4` 옵션으로 클립을 번호별로 나누어 병렬 처리할 수 있습니다.  
- **짝홀 분할**: `--odd-only`, `--even-only` 옵션은 클러스터 자원을 두 그룹으로 나눠 쓸 때 유용했습니다.  
- **단일 파일 처리**: `--video`와 `--output`을 지정하면 한 파일에 대한 추론과 JSON 생성을 빠르게 테스트할 수 있습니다.  

## 견고함을 위한 설계 포인트
- **사전 검증 단계**  
  - 파일 크기, 헤더, ffprobe 결과를 점검해 문제가 있는 영상은 자동으로 `repair_video` 단계로 넘깁니다.  
  - ffmpeg 재패키징으로 해결되지 않으면 별도의 큐에 등록해 수동 검토 대상으로 분류합니다.
  - 폴백 시에도 JSON 스키마를 맞추기 위해 키포인트 항목을 빈 배열로 유지하여 후속 처리에서 일관성을 보장합니다.
- **FrameAccumulator**  
  - 프레임별 결과를 메모리에 모았다가 주기적으로 JSON과 DB에 일괄 저장해 IO 횟수를 낮췄습니다.  
  - 메모리 사용량이 일정 임계치를 넘으면 자동으로 flush가 일어나도록 안전장치를 추가했습니다.
- **재실행 지능**  
  - JSON이 이미 존재하면 스킵하고, DB에는 `processed` 플래그를 남겨 중복 삽입을 방지합니다.  
  - CLI 실행 결과는 `annotations/manifest.tsv`에 기록되어, 언제 어떤 옵션으로 처리했는지 추적할 수 있습니다.

## 핵심 데이터 스키마
- `videos` 테이블: 파일명, 총 프레임, FPS, 해상도, 처리 시간, JSON 경로, 처리 상태 등을 기록합니다.
- `detections` 테이블: 프레임 번호, 얼굴 ID, 바운딩 박스 좌표, confidence, 5개 키포인트를 저장합니다.
- JSON 구조: `video_info` 메타데이터와 `frames.frame_{n}` 객체로 구성하며, 각 프레임마다 `detections` 배열을 포함합니다.

## 데이터 가공 기준 반영 사항
- 전처리 표준: 해상도 가로·세로 1/2 축소, 150초(≈2분 30초) 세그먼트 분할을 기본값으로 사용합니다.
- 라벨링 규격: 프레임당 바운딩 박스와 5점 특징점(좌/우 눈, 코, 좌/우 입꼬리) 좌표를 JSON에 기록합니다.
- 무결성 검증: 완전성/유일성/유효성/일관성/정확성 기준으로 JSON 스키마 검증을 통과해야 후속 단계로 진행됩니다.
- 품질 지표: 구조 오류율 0.1% 미만을 목표로 하며, 재처리 큐를 통해 자동/수동 보정 루틴을 운용합니다.
- 산출물 체계: 영상 MP4와 동일 파일명으로 1:1 매칭되는 JSON을 생성하고, `annotations/manifest.tsv`에 처리 이력을 남깁니다.

## 운영에서 나온 베스트 프랙티스
- **가중치 준비**: YOLOv8-face 전용 가중치가 없을 때는 `yolov8n.pt`로 폴백하지만, 키포인트 품질이 떨어지므로 배포 체크리스트에 가중치 다운로드 절차를 반드시 포함했습니다.
- **리소스 제어**: `--max-dim` 값을 낮추면 GPU 메모리를 절약할 수 있고, CPU 환경에서는 `--device cpu` + `--batch-size 1` 조합이 가장 안정적이었습니다.
- **배치 처리**: `scripts/batch-process-safe.sh`와 조합하면 번호 단위 파일 락을 활용해 4~6개의 워커를 동시에 돌리는 것이 가장 효율적이었습니다.
- **로그 관리**: 실패한 프레임은 별도 로그에 기록해, 나중에 `preview_annotations.py`로 시각화하며 원인을 분석했습니다.

## 사례: 데이터 가공 용역 기반 처리
- 수요기업 제공 고화질 원본 영상을 해상도 1/2로 축소하고 150초 단위로 분할해 전처리했습니다.  
- 총 3,000건의 MP4 영상과 3,000건의 JSON 어노테이션을 1:1로 구축했습니다.  
- 프레임별 얼굴 바운딩 박스와 5점 특징점 좌표를 라벨링하고, JSON 스키마 무결성 검증을 통과한 결과만 반영했습니다.  
- 교차검증을 포함한 2차 라벨링을 통해 구조 오류율 0.1% 미만을 달성했습니다.    
- 추진 일정: 2025-06-01~06-15 기획/설계 → 06-16~07-06 정제/증강 → 07-07~07-27 1차 라벨링 → 07-28~08-17 2차 라벨링/교차검증 → 08-18~08-31 최종 검수/납품.

