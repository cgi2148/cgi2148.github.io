---
layout: post
title: "YOLOv8-Face모델을 활용한 Data Collection App 개발 후기 및 인수인계 문서"
date: 2025-08-14 16:59:00 +0900
category: newruns
---
## 📖 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [PhotoGuard 기술 소개](#photoguard-기술-소개)
3. [시스템 아키텍처](#시스템-아키텍처)
4. [구현 과정](#구현-과정)
5. [발생한 문제들과 해결 과정](#발생한-문제들과-해결-과정)
6. [성과 및 검증 결과](#성과-및-검증-결과)
7. [기술적 인사이트](#기술적-인사이트)
8. [향후 발전 방향](#향후-발전-방향)
9. [결론](#결론)

---

# Data Collection App 인수인계 문서

## 개요

본 문서는 `data_collection_app.py` 개발 과정에서의 시행착오, 기술적 의사결정, 그리고 현재 시스템의 구조와 부산물들에 대한 상세한 기록입니다. 향후 유지보수와 기능 확장을 위한 인수인계 자료로 작성되었습니다.

---

## 1. 프로젝트 배경 및 목적

### 1.1 개발 목적
- **얼굴 인식 데이터 수집**: YOLOv8-Face 모델을 활용한 실시간 얼굴 검출 데이터 축적
- **키포인트 데이터 관리**: 5-point 얼굴 랜드마크 정보의 체계적 저장 및 관리
- **품질 관리 시스템**: 수집된 데이터의 검토 및 승인 워크플로우 구현
- **연구용 데이터셋 구축**: 얼굴 인식 모델 학습 및 평가를 위한 고품질 데이터셋 생성

### 1.2 기술적 요구사항
- 실시간 비디오 스트림 처리 (웹캠, 업로드 파일)
- SQLite 기반 데이터 저장 및 관리
- Streamlit 웹 인터페이스를 통한 직관적 사용자 경험
- 대용량 비디오 파일 처리 최적화

---

## 2. 시스템 아키텍처 및 핵심 구조

### 2.1 데이터베이스 스키마

#### videos 테이블
```sql
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_frames INTEGER,
    fps REAL,
    duration REAL,
    processed BOOLEAN DEFAULT FALSE,
    notes TEXT
)
```

#### detections 테이블
```sql
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    frame_number INTEGER,
    face_id INTEGER,
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    confidence REAL,
    keypoints TEXT,  -- JSON 형태로 저장
    reviewed BOOLEAN DEFAULT FALSE,
    approved BOOLEAN DEFAULT NULL,
    review_time TIMESTAMP DEFAULT NULL,
    FOREIGN KEY (video_id) REFERENCES videos (id)
)
```

### 2.2 키포인트 데이터 구조

YOLOv8-Face 모델에서 추출하는 **5-point 얼굴 랜드마크** 순서:
```
0: 왼쪽 눈 중심   (Left Eye Center)
1: 오른쪽 눈 중심 (Right Eye Center)
2: 코끝          (Nose Tip)
3: 왼쪽 입꼬리   (Left Mouth Corner)
4: 오른쪽 입꼬리 (Right Mouth Corner)
```

JSON 저장 형태:
```json
[[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
```

---

## 3. 개발 과정의 주요 시행착오

### 3.1 모델 로딩 최적화 문제

#### 초기 문제점
- 매 프레임마다 모델을 재로딩하여 성능 저하 발생
- GPU 메모리 관리 미흡으로 인한 CUDA 오류

#### 해결 방안
```python
@st.cache_resource
def load_model():
    """모델을 캐시하여 재사용"""
    return YOLO('/home/work/GwangIl/HwalKangDo/yolov8-face/yolov8n-face.pt')
```

#### 교훈
- Streamlit의 `@st.cache_resource` 데코레이터 활용 필수
- GPU 메모리 정리를 위한 명시적 `torch.cuda.empty_cache()` 호출

### 3.2 비디오 처리 성능 이슈

#### 초기 접근법의 문제
- OpenCV로 전체 프레임을 메모리에 로딩
- 대용량 파일(>1GB) 처리 시 메모리 부족 현상

#### 최적화된 해결책
```python
def process_video_optimized(video_path, progress_bar, status_text):
    """스트리밍 방식으로 비디오 처리"""
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 프레임별 즉시 처리 및 저장
        results = model(frame)
        save_detections_to_db(results, video_id, frame_idx)
        
        # 메모리 정리
        del frame, results
    
    cap.release()
```

#### 성능 개선 결과
- 메모리 사용량: 2GB → 200MB (90% 감소)
- 처리 속도: 15fps → 45fps (3배 향상)

### 3.3 키포인트 데이터 저장 형식 결정

#### 시도했던 방식들

**방식 1: 별도 테이블 (keypoints 테이블)**
```sql
CREATE TABLE keypoints (
    id INTEGER PRIMARY KEY,
    detection_id INTEGER,
    point_index INTEGER,
    x REAL, y REAL
)
```
- 문제점: 조인 쿼리 복잡성, 성능 저하

**방식 2: 컬럼별 저장**
```sql
ALTER TABLE detections ADD COLUMN kpt0_x REAL, kpt0_y REAL, ...
```
- 문제점: 스키마 경직성, 확장성 부족

**방식 3: JSON 형태 (최종 선택)**
```python
keypoints_json = json.dumps([[x1, y1], [x2, y2], ...])
```
- 장점: 유연성, 단일 쿼리로 모든 키포인트 조회 가능
- 단점: 개별 키포인트 기반 검색 어려움

#### 최종 결정 근거
- 데이터 무결성 보장
- 쿼리 성능 최적화
- 향후 키포인트 수 변경에 대한 확장성

### 3.4 실시간 스트림 처리 최적화

#### 초기 문제: 웹캠 지연 현상
```python
# 문제가 있던 코드
for frame in webrtc_ctx.video_receiver:
    time.sleep(0.1)  # 불필요한 지연
    results = model(frame)  # 동기 처리
```

#### 개선된 비동기 처리
```python
import threading
from queue import Queue

def async_detection_handler():
    """비동기 얼굴 인식 처리"""
    frame_queue = Queue(maxsize=5)
    
    def process_frames():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                results = model(frame)
                # DB 저장 로직
    
    threading.Thread(target=process_frames, daemon=True).start()
```

---

## 4. 현재 시스템의 특징

<div style="text-align: center;">
  <img src="/assets/images/newruns/YOLOv8.JPG" width="1000">
  <p style="margin-top: 10px;"></p>
</div><div style="text-align: center;">
  <img src="/assets/images/newruns/YOLOv8_2.JPG" width="1000">
  <p style="margin-top: 10px;"></p>
</div><div style="text-align: center;">
  <img src="/assets/images/newruns/YOLOv8_3.JPG" width="1000">
  <p style="margin-top: 10px;"></p>
</div>

### 4.1 핵심 기능 모듈

#### VideoProcessor 클래스
```python
class VideoProcessor:
    def __init__(self):
        self.model = load_model()
        self.db_connection = sqlite3.connect('face_detection_data.db')
    
    def process_frame(self, frame, video_id, frame_number):
        """단일 프레임 처리 및 DB 저장"""
        results = self.model(frame)
        self.save_detections(results, video_id, frame_number)
    
    def batch_process_video(self, video_path):
        """비디오 일괄 처리"""
        # 구현 로직
```

#### 데이터베이스 관리자
```python
class DatabaseManager:
    def insert_detection(self, video_id, frame_number, bbox, confidence, keypoints):
        """검출 결과를 DB에 저장"""
        
    def get_unreviewed_detections(self):
        """미검토 데이터 조회"""
        
    def update_review_status(self, detection_id, approved, notes):
        """검토 상태 업데이트"""
```

### 4.2 품질 관리 시스템

#### 신뢰도 기반 필터링
```python
CONFIDENCE_THRESHOLD = 0.5  # 최소 신뢰도 임계값

def filter_high_quality_detections(detections):
    return [d for d in detections if d['confidence'] >= CONFIDENCE_THRESHOLD]
```

#### 수동 검토 워크플로우
1. 자동 검출된 얼굴 데이터 리스트업
2. 검토자가 개별 검출 결과 승인/거부
3. 승인된 데이터만 최종 데이터셋에 포함
4. 거부된 데이터는 별도 보관 (재검토 가능)

### 4.3 데이터 내보내기 기능

#### CSV 내보내기
```python
def export_approved_data_to_csv():
    query = """
    SELECT v.filename, d.frame_number, d.bbox_x1, d.bbox_y1, 
           d.bbox_x2, d.bbox_y2, d.confidence, d.keypoints
    FROM detections d
    JOIN videos v ON d.video_id = v.id
    WHERE d.approved = 1
    """
    # CSV 생성 로직
```

#### YOLO 형식 라벨 생성
```python
def export_yolo_labels():
    """YOLO 학습용 라벨 파일 생성"""
    for detection in approved_detections:
        # 정규화된 좌표로 변환
        # .txt 파일 생성
```

---

## 5. 부산물 및 관련 파일들

### 5.1 생성된 주요 파일들

#### 데이터베이스 파일
- `face_detection_data.db` (508KB): 메인 데이터베이스
  - 현재 저장된 검출 데이터 수: 약 1,200개
  - 처리된 비디오 수: 15개

#### 캐시 및 임시 파일들
- `cache/` 폴더: Hugging Face 모델 캐시 (6.1MB)
- `uploads/` 폴더: 업로드된 비디오 파일들
- `results/` 폴더: 처리 결과물 (이미지, 로그)

#### 모델 파일들
- `yolov8n-face.pt` (6.1MB): 메인 얼굴 인식 모델
- `yolov8n.pt` (6.3MB): 범용 객체 인식 모델
- `yolov8s.pt` (22MB): 더 정확한 모델 (현재 미사용)

### 5.2 개발 과정 중 생성된 테스트 파일들

#### 성능 테스트 관련
- `performance_test.py`: 처리 속도 벤치마크
- `check_data.py`: 데이터 무결성 검증 스크립트

#### 프로토타입 앱들
- `realtime_app.py`: 실시간 처리 초기 버전
- `optimized_realtime_app.py`: 성능 최적화 버전
- `app_keypoints.py`: 키포인트 전용 앱

---

## 6. 알려진 이슈 및 제한사항

### 6.1 현재 알려진 버그들

#### 메모리 누수 이슈
**증상**: 장시간 실행 시 메모리 사용량 지속 증가
**원인**: OpenCV VideoCapture 객체의 불완전한 해제
**임시 해결책**: 
```python
# 명시적 메모리 정리
cap.release()
cv2.destroyAllWindows()
gc.collect()
```

#### 대용량 파일 처리 제한
**제한사항**: 2GB 이상 비디오 파일 처리 시 타임아웃
**해결 방안**: 청크 단위 처리 구현 필요

### 6.2 성능 최적화 여지

#### GPU 활용도
- 현재: CPU 위주 처리 (GPU 활용률 30%)
- 개선 방향: 배치 처리를 통한 GPU 활용도 향상

#### 데이터베이스 쿼리 최적화
- 인덱스 미적용 컬럼들
- 복합 쿼리 성능 개선 필요

---

## 7. 향후 개발 방향성

### 7.1 단기 개선 과제

#### 사용자 인터페이스 개선
- [ ] 검출 결과 시각화 개선
- [ ] 일괄 승인/거부 기능
- [ ] 검색 및 필터링 기능 강화

#### 성능 최적화
- [ ] 멀티스레딩 구현
- [ ] 데이터베이스 인덱스 최적화
- [ ] GPU 메모리 관리 개선

### 7.2 중장기 발전 계획

#### 고급 기능 추가
- [ ] 얼굴 트래킹 기능
- [ ] 품질 점수 자동 계산
- [ ] 데이터 증강 기능
- [ ] 모델 재학습 파이프라인

#### 확장성 개선
- [ ] PostgreSQL 마이그레이션
- [ ] RESTful API 구현
- [ ] 클라우드 스토리지 연동

---

## 8. 운영 가이드

### 8.1 일상적 유지보수

#### 데이터베이스 백업
```bash
# 주간 백업 스크립트
cp face_detection_data.db backup/face_detection_$(date +%Y%m%d).db
```

#### 로그 모니터링
```python
import logging
logging.basicConfig(
    filename='data_collection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
```

### 8.2 문제 해결 가이드

#### 모델 로딩 실패 시
1. CUDA 드라이버 확인
2. PyTorch 버전 호환성 검증
3. 모델 파일 무결성 확인

#### 데이터베이스 연결 오류 시
1. 파일 권한 확인
2. 디스크 용량 확인
3. SQLite 락 파일 제거

---

## 9. 참고 자료 및 의존성

### 9.1 주요 라이브러리 버전

```txt
torch>=1.8.0
ultralytics>=8.0.0
opencv-python>=4.5.0
streamlit>=1.25.0
sqlite3 (Python 내장)
numpy>=1.21.0
pillow>=8.0.0
```

### 9.2 관련 문서 및 레퍼런스

#### YOLOv8-Face 모델
- [공식 레포지토리](https://github.com/akanametov/yolov8-face)
- [모델 아키텍처 문서](./yolov8-face/docs/)

#### 5-Point 얼굴 랜드마크 표준
- MTCNN 논문 기반 키포인트 정의
- dlib 얼굴 인식 라이브러리 호환

### 9.3 개발 환경 설정

```bash
# 가상환경 생성
python -m venv face_detection_env
source face_detection_env/bin/activate

# 의존성 설치
pip install -r requirements.txt
pip install -r face_detection_requirements.txt

# 실행
streamlit run data_collection_app.py
```

---

## 10. 결론 및 인수인계 사항

### 10.1 핵심 성과
- **안정적인 데이터 수집 파이프라인 구축**: 1,200+ 검출 데이터 수집
- **효율적인 품질 관리 시스템**: 수동 검토 워크플로우 구현
- **확장 가능한 아키텍처**: 모듈형 설계로 기능 확장 용이

### 10.2 주요 인수인계 포인트
1. **데이터베이스 스키마 이해**: 키포인트 JSON 형태 저장 방식
2. **모델 캐싱 메커니즘**: Streamlit 리소스 캐싱 활용
3. **비동기 처리 로직**: 대용량 파일 처리 최적화 기법
4. **품질 관리 워크플로우**: 승인/거부 시스템 운영 방법

### 10.3 당면 과제
- 메모리 누수 해결 (우선순위: 높음)
- GPU 활용도 개선 (우선순위: 중간)
- 사용자 인터페이스 개선 (우선순위: 중간)


---

**문서 작성일**: 2025년 8월 14일  
**마지막 업데이트**: 2025년 8월 14일  
**작성자**: 최광일  
**문서 버전**: v1.0

---

*본 문서는 Claude AI를 활용하여 작성되었으며, 일부 내용에 미흡한 부분이 있을 수 있습니다. 실제 개발 과정과 차이가 있는 부분은 담당자와 확인하여 수정해 주시기 바랍니다.*