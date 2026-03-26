# 카메라 낙상감지 파이프라인 컨텍스트

Claude에게 넘겨줄 때 이 파일을 첨부하면 됨.

---

## 개요

YOLOv8-Pose + LSTM(FallNet)으로 카메라 기반 낙상 감지.
WiFi CSI와 별개로 동작하는 독립 파이프라인.

- **클래스**: fall / normal (2-class)
- **키포인트**: YOLOv8n-pose → 17 keypoint × 2(x,y) = 34 features
- **모델**: Bidirectional LSTM(128) → Linear(256→128) → Linear(128→2)
- **윈도우**: 30 프레임, stride 5

---

## 현재 상태 (2026-03-26)

### 최신 모델
- 파일: `v1/data/models/fall_camera_20260326_144930.pt`
- val 정확도: 99.1%
- 학습 데이터: AI Hub fall 3321 윈도우 + 직접 촬영 normal 323 윈도우

### 문제점
- **오감지 발생**: normal 데이터(323개)가 fall 데이터(3321개)에 비해 너무 적어서 서 있어도 fall로 판정하는 경우 있음
- **해결 방법**: AI Hub N(정상) 데이터 추가 학습 필요
- confidence 임계값 현재 0.90으로 올려놓음

---

## PC에서 이어하기 (Windows + RTX 5060Ti)

### 1. 환경 세팅
```bash
pip install ultralytics torch torchvision
git clone https://github.com/b-hyoung/RuView
cd RuView
```

### 2. AI Hub N(정상) 데이터 받기
- AI Hub → 041.낙상사고 위험동작 영상-센서 쌍 데이터
- **TS.z01** (100GB, key: 531131) 다운로드 → 외장하드에 압축 해제
- 압축 해제 후 `이미지/N/N/` 폴더 안에 JPG 이미지들 있어야 함

### 3. N 데이터 키포인트 추출
`v1/extract_aihub_keypoints.py` 상단 수정:
```python
IMAGE_ROOT = "D:/경로/이미지/N/N"   # Windows 경로로 수정
OUTPUT     = "v1/data/camera_dataset/aihub_normal_dataset.json"
```
그리고 `main()` 안에서 label을 `"normal"`로 바꿔야 함:
```python
data = {"data": {"fall": [], "normal": fall_windows}}  # fall_windows → normal
```
실행:
```bash
python v1/extract_aihub_keypoints.py
```

### 4. AI Hub fall 데이터 추출 (외장하드 Y 폴더)
```python
IMAGE_ROOT = "D:/경로/이미지/Y"   # 또는 외장하드 경로
OUTPUT     = "v1/data/camera_dataset/aihub_fall_dataset.json"
```
```bash
python v1/extract_aihub_keypoints.py
```

### 5. 재학습
```bash
python v1/train_camera_model.py \
  --dataset v1/data/camera_dataset/aihub_fall_dataset.json \
  --dataset v1/data/camera_dataset/aihub_normal_dataset.json \
  --dataset v1/data/camera_dataset/cam_session_20260326_103253_dataset.json
```

### 6. 실시간 추론
```bash
python v1/camera_inference.py --model v1/data/models/fall_camera_XXXXXXXX.pt --camera 0
```
카메라 인덱스는 `--camera 0` 또는 `--camera 1` (어떤 게 맞는지 테스트)

---

## 스크립트 설명

| 파일 | 역할 |
|------|------|
| `v1/collect_camera_data.py` | 카메라로 학습 데이터 직접 촬영 (맥 전용) |
| `v1/extract_camera_keypoints.py` | 직접 찍은 영상 → YOLO 키포인트 추출 |
| `v1/extract_aihub_keypoints.py` | AI Hub 이미지 폴더 → YOLO 키포인트 추출 |
| `v1/train_camera_model.py` | FallNet LSTM 학습 (--dataset 여러 개 가능) |
| `v1/camera_inference.py` | 실시간 낙상 감지 (카메라) |
| `v1/realtime_inference.py` | WiFi CSI 기반 실시간 추론 (HTTP polling) |

---

## 낙상 감지 파라미터 (camera_inference.py)

```python
FALL_CONFIRM_SEC = 0.5   # fall 판정 유지해야 하는 시간(초)
ALERT_COOLDOWN   = 30.0  # 알람 재울림 방지 쿨다운(초)
FALL_TOLERANCE   = 8     # fall 중간에 끊겨도 허용할 프레임 수
conf > 0.90              # fall 판정 confidence 임계값
```

---

## 데이터셋 구조 (JSON)

```json
{
  "data": {
    "fall": [[...34 floats × 30 frames...], ...],
    "normal": [[...34 floats × 30 frames...], ...]
  }
}
```
