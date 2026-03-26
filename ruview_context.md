# RuView 프로젝트 컨텍스트

## 프로젝트 개요
- **RuView**: WiFi DensePose 기반 실시간 인체 자세 추정 시스템
- **GitHub**: https://github.com/b-hyoung/RuView (원본: ruvnet/RuView)
- **원리**: WiFi 신호의 CSI(Channel State Information)를 분석해 카메라 없이 사람의 위치, 자세, 생체신호(심박/호흡) 감지
- **핵심 기술**: ESP32-S3가 WiFi 신호 산란 패턴을 캡처 → AI 분석 → 17개 신체 keypoint 추정

---

## 하드웨어 환경
- **ESP32 보드**: ESP32-S3 개발보드 N16R8 (ESPRESSIF WROOM-1) × 3개
  - USB-C 포트 탑재 (Micro-USB 아님)
  - 16MB Flash, 8MB PSRAM
  - 맥북 C to C 케이블로 바로 연결 가능
- **맥북**: MacBook Pro M4 Pro (Apple Silicon)
- **WiFi**: iptime5G (비밀번호 없음)
- **맥북 IP**: 192.168.0.167

---

## 소프트웨어 환경
- **OS**: macOS (Apple Silicon)
- **Python**: python3
- **esptool**: v5.2.0 (설치 완료)
- **Docker**: v29.1.2 (설치 완료)
- **RuView 경로**: `~/bobs_project/RuView` (iCloud Drive에 위치)
- **펌웨어 경로**: `~/bobs_project/RuView/firmware/esp32-csi-node`

---

## 현재 진행 상황

### 완료된 것 ✅
1. ESP32-S3 맥북 USB-C로 연결 및 인식 확인
   - 포트: `/dev/cu.usbmodem5B8E0626371`
2. `provision.py`로 WiFi 설정 완료
   - SSID: iptime5G
   - Target IP: 192.168.0.167
   - Node ID: 1
3. Docker로 RuView 서버 실행 완료
   - `docker run -p 3000:3000 -p 3001:3001 -p 5005:5005/udp ruvnet/wifi-densepose:latest`
   - 대시보드: http://localhost:3000
4. 시뮬레이션 모드로 자세 감지 UI 확인 완료

### 남은 것 ⏳
- **ESP32 펌웨어 빌드 및 플래시** (ESP-IDF 툴체인 설치 필요)
  - 펌웨어 소스: `firmware/esp32-csi-node`
  - 빌드 도구: ESP-IDF
  - 빌드 후 플래시하면 실제 CSI 데이터 수집 시작
- ESP32 3개 모두 펌웨어 올리기 (현재 1개만 WiFi 설정됨)
- 실제 CSI 데이터로 자세 감지 테스트

---

## 주요 명령어 모음

### ESP32 포트 확인
```bash
ls /dev/cu.*
# 결과: /dev/cu.usbmodem5B8E0626371
```

### WiFi 프로비저닝
```bash
cd ~/bobs_project/RuView/firmware/esp32-csi-node
python3 provision.py \
  --port /dev/cu.usbmodem5B8E0626371 \
  --ssid "iptime5G" \
  --target-ip 192.168.0.167 \
  --node-id 1
# node-id는 각 ESP32마다 1, 2, 3으로 다르게 설정
```

### Docker 서버 실행
```bash
docker run -p 3000:3000 -p 3001:3001 -p 5005:5005/udp ruvnet/wifi-densepose:latest
```

### 대시보드 접속
```
http://localhost:3000
```

---

## 다음 단계: ESP-IDF 펌웨어 빌드

### ESP-IDF 설치 (Apple Silicon)
```bash
mkdir -p ~/esp
cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32s3
. ./export.sh
```

### 펌웨어 빌드 및 플래시
```bash
cd ~/bobs_project/RuView/firmware/esp32-csi-node
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/cu.usbmodem5B8E0626371 flash
```

---

## 아키텍처 요약
```
ESP32-S3 × 3개 (CSI 수집)
    ↓ WiFi UDP (port 5005)
맥북 Docker 서버 (RuView)
    ↓
http://localhost:3000 (대시보드)
```

---

## 카메라 낙상감지
→ 자세한 내용: `docs/camera-fall-detection/context.md`

---

## 참고 링크
- 라이브 데모: https://ruvnet.github.io/RuView/
- 포즈 퓨전 데모: https://ruvnet.github.io/RuView/pose-fusion.html
- GitHub: https://github.com/b-hyoung/RuView
