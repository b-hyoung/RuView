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

## 현재 상태 (2026-03-27)

### 최신 모델
- `v1/data/models/` 안에 가장 최근 `fall_camera_YYYYMMDD_HHMMSS.pt` 사용
- 학습 데이터:
  - AI Hub fall (BY+FY+SY 전체) — 수만 개 윈도우
  - AI Hub normal (N/N 전체) — 수만 개 윈도우
  - 직접 촬영 normal 323 윈도우
- Windows PC(RTX 5060Ti)에서 재학습 완료 → git push 됨

### 이전 문제점 (해결됨)
- normal 데이터 323개 vs fall 수천 개 → 심각한 불균형으로 오감지 발생
- AI Hub N 데이터 대규모 추가로 해결

### 맥에서 테스트
```bash
git pull origin main
ls v1/data/models/   # 가장 최근 .pt 파일 확인
python3 v1/camera_inference.py --model v1/data/models/fall_camera_최신파일.pt --camera 0
```
- 카메라 인덱스 `--camera 0` 또는 `--camera 1` 테스트
- confidence 임계값 현재 0.90

---

## 추가 재학습이 필요한 경우 (Windows PC)

`run_all.bat` 실행하면 Fall 추출 → 학습 → git push 자동:
```cmd
run_all.bat
```

`extract_aihub_keypoints.py`는 `--mode` 인자로 동작:
```bash
python v1/extract_aihub_keypoints.py --mode normal  # N 데이터
python v1/extract_aihub_keypoints.py --mode fall    # Y 데이터 (BY+FY+SY)
```
데이터 경로: `D:/ai_nak/041.낙상사고.../TS/이미지/`

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
