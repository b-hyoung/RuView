#!/usr/bin/env python3
"""
Camera Fall Detection - Realtime Inference
맥캠 → YOLOv8-Pose → LSTM → 낙상 감지 + 삐삐

사용법:
  python3 v1/camera_inference.py --model v1/data/models/fall_camera_XXXXXXXX.pt
"""

import cv2, json, argparse, time, threading, subprocess
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from ultralytics import YOLO

WINDOW   = 30
FEATURES = 34  # 17 kp × 2
LABELS   = ["fall", "normal"]

FALL_CONFIRM_SEC = 0.5   # fall 판정 유지 시간
ALERT_COOLDOWN   = 30.0
FALL_TOLERANCE   = 8     # fall 중간에 몇 프레임 끊겨도 허용

# ── Model ─────────────────────────────────────────────────────────────────────

class FallNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, 128, num_layers=2, batch_first=True,
                            dropout=0.0, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

# ── Fall alert ────────────────────────────────────────────────────────────────

def trigger_alert():
    print("\n" + "="*50)
    print("🚨  FALL DETECTED!  낙상 감지!  🚨")
    print("="*50)
    for _ in range(3):
        subprocess.Popen(["afplay", "/System/Library/Sounds/Sosumi.aiff"])
        time.sleep(0.6)
    subprocess.Popen(["say", "-v", "Yuna", "낙상이 감지되었습니다. 확인이 필요합니다."])

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", type=int, default=1, help="카메라 인덱스 (기본값 1)")
    args = parser.parse_args()

    meta_path = args.model.replace(".pt", "_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    labels = meta["labels"]
    n_classes = len(labels)

    # Load models
    print("Loading YOLO...")
    yolo = YOLO("yolov8n-pose.pt")

    print("Loading FallNet...")
    fall_net = FallNet(n_classes)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    fall_net.load_state_dict(state)
    fall_net.eval()
    print(f"Ready. Labels: {labels}")

    # Frame buffer
    frame_buf = deque(maxlen=WINDOW)

    # Fall state
    fall_since     = None
    last_alert     = 0.0
    last_label     = "normal"
    non_fall_count = 0   # fall 아닌 프레임 연속 카운트

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("ERROR: 카메라를 열 수 없습니다. 카메라 권한을 확인하세요.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # macOS 카메라 워밍업
    import time
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    print("Running... Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: 프레임을 읽을 수 없습니다.")
            break

        # 1. Extract keypoints
        results = yolo(frame, verbose=False)
        person_detected = False
        kp = [0.0] * FEATURES
        if results and results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
            kp_raw = results[0].keypoints.xyn[0].cpu().numpy()
            # 유효 키포인트(0이 아닌 것) 15개 이상이어야 전신으로 인정
            valid = np.count_nonzero(kp_raw.sum(axis=1))
            if valid >= 8:
                kp = kp_raw.flatten().tolist()
                person_detected = True
        frame_buf.append(kp)

        # 2. Classify when buffer full
        label, conf = last_label, 0.0
        if len(frame_buf) == WINDOW:
            x = torch.tensor([list(frame_buf)], dtype=torch.float32)
            with torch.no_grad():
                logits = fall_net(x)
            probs     = torch.softmax(logits[0], dim=0).numpy()
            label_idx = int(probs.argmax())
            label     = labels[label_idx]
            conf      = float(probs[label_idx])
            last_label = label

        # 3. Fall detection state machine (tolerance: 중간에 끊겨도 허용)
        now = time.time()
        if not person_detected:
            label, conf = "normal", 0.0
        if label == "fall" and conf > 0.90:
            non_fall_count = 0
            if fall_since is None:
                fall_since = now
            elif now - fall_since >= FALL_CONFIRM_SEC:
                if now - last_alert >= ALERT_COOLDOWN:
                    last_alert = now
                    threading.Thread(target=trigger_alert, daemon=True).start()
        else:
            non_fall_count += 1
            if non_fall_count >= FALL_TOLERANCE:
                fall_since = None
                non_fall_count = 0

        # 4. Draw overlay
        color = {"fall": (0,0,220), "normal": (0,200,0)}.get(label, (200,200,200))
        icon  = "🚨" if label == "fall" else "✓"

        # Draw YOLO skeleton
        annotated = results[0].plot() if results else frame

        # Status box
        cv2.rectangle(annotated, (0,0), (320,60), (0,0,0), -1)
        cv2.putText(annotated, f"{label.upper()}  {conf*100:.0f}%",
                    (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        if label == "fall":
            elapsed = now - fall_since if fall_since else 0
            cv2.putText(annotated, f"FALL {elapsed:.1f}s / {FALL_CONFIRM_SEC:.0f}s",
                        (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,220), 1)

        cv2.imshow("Fall Detection", annotated)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
