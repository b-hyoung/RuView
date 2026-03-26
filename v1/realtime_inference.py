#!/usr/bin/env python3
"""
RuView realtime inference + 낙상 감지
활동 분류 → 낙상 패턴 감지 → 삐삐 알림

사용법:
  python3 v1/realtime_inference.py --model v1/data/models/csi_pose_raw_20260325_172709.pt
"""

import json, argparse, time, threading, requests, subprocess
import numpy as np
import torch
import torch.nn as nn
from collections import deque, defaultdict

SERVER   = "http://localhost:8181"
POLL_HZ  = 10
N_SUB    = 50
N_NODES  = 3
IN_DIM   = N_SUB * N_NODES
WINDOW   = 30

# 낙상 감지 설정
FALL_CONFIRM_SEC = 4.0   # 누워있는 상태가 4초 이상 → 낙상
ALERT_COOLDOWN   = 30.0  # 30초마다 최대 1번 알림
UPRIGHT_LABELS   = {"standing", "sitting", "walking", "gestures", "sit_stand"}
FALL_LABEL       = "lying"
MIN_CONF         = 0.60  # 신뢰도 60% 미만은 무시

# ── Model ─────────────────────────────────────────────────────────────────────

class CsiPoseNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(IN_DIM, 256, num_layers=2, batch_first=True,
                            dropout=0.0, bidirectional=True)
        self.cls_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
        self.kp_head = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 99), nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        feat = out[:, -1, :]
        return self.cls_head(feat), self.kp_head(feat)

# ── 낙상 감지 상태 머신 ────────────────────────────────────────────────────────

class FallDetector:
    def __init__(self):
        self.state       = "normal"   # normal | candidate | fallen
        self.lying_since = None
        self.last_alert  = 0.0

    def update(self, label: str, conf: float) -> bool:
        """낙상이 새로 감지되면 True 반환"""
        now = time.time()

        if label == FALL_LABEL and conf >= MIN_CONF:
            if self.state == "normal":
                self.state = "candidate"
                self.lying_since = now
                print(f"\n[낙상 감지 대기] 누워있음 감지... {FALL_CONFIRM_SEC:.0f}초 확인 중")
            elif self.state == "candidate":
                elapsed = now - self.lying_since
                if elapsed >= FALL_CONFIRM_SEC:
                    if now - self.last_alert >= ALERT_COOLDOWN:
                        self.state = "fallen"
                        self.last_alert = now
                        return True  # 낙상 확정!

        elif label in UPRIGHT_LABELS and conf >= MIN_CONF:
            if self.state in ("candidate", "fallen"):
                print(f"\n[낙상 해제] 일어남 감지")
            self.state = "normal"
            self.lying_since = None

        return False

def trigger_alert():
    """macOS 삐삐 + 음성 알림"""
    print("\n" + "="*50)
    print("🚨  낙상 감지!  FALL DETECTED!  🚨")
    print("="*50)
    # 삐삐 3번
    for _ in range(3):
        subprocess.Popen(["afplay", "/System/Library/Sounds/Sosumi.aiff"])
        time.sleep(0.6)
    # 음성 (macOS Yuna 한국어 목소리)
    subprocess.Popen(["say", "-v", "Yuna", "낙상이 감지되었습니다. 확인이 필요합니다."])

# ── Inference engine ──────────────────────────────────────────────────────────

class RealtimeInference:
    def __init__(self, model_path, meta_path):
        with open(meta_path) as f:
            self.meta = json.load(f)
        self.labels = self.meta["labels"]
        self.model  = CsiPoseNet(len(self.labels))
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        print(f"Model loaded: {self.labels}")

        self.node_buf  = defaultdict(lambda: deque(maxlen=WINDOW*2))
        self.last_pred = {"label": "unknown", "confidence": 0.0, "keypoints": []}
        self.lock      = threading.Lock()
        self.fall_det  = FallDetector()

    def push_nodes(self, nodes: list):
        with self.lock:
            for node in nodes:
                nid = node.get("node_id", 0)
                amp = node.get("amplitude", [])[:N_SUB]
                amp = amp + [0.0] * (N_SUB - len(amp))
                self.node_buf[nid].append(amp)
            self._maybe_infer()

    def _maybe_infer(self):
        if len(self.node_buf) < 2:
            return
        min_len = min(len(v) for v in self.node_buf.values())
        if min_len < WINDOW:
            return

        node_ids = sorted(self.node_buf.keys())[:N_NODES]
        frames = []
        for t in range(-WINDOW, 0):
            row = []
            for nid in node_ids:
                buf = list(self.node_buf[nid])
                row.extend(buf[t] if abs(t) <= len(buf) else [0.0]*N_SUB)
            while len(row) < IN_DIM:
                row.extend([0.0] * N_SUB)
            frames.append(row[:IN_DIM])

        x = torch.tensor([frames], dtype=torch.float32) / 20.0
        with torch.no_grad():
            logits, kp_out = self.model(x)
        probs     = torch.softmax(logits[0], dim=0).numpy()
        label_idx = int(probs.argmax())
        label     = self.labels[label_idx]
        if hasattr(label, 'item'):
            label = label.item()
        conf = float(probs[label_idx])
        kps  = kp_out[0].numpy().tolist()

        self.last_pred = {"label": label, "confidence": conf, "keypoints": kps}

        # 낙상 감지 체크
        if self.fall_det.update(label, conf):
            threading.Thread(target=trigger_alert, daemon=True).start()

    def get_prediction(self):
        with self.lock:
            return dict(self.last_pred)

# ── HTTP polling ──────────────────────────────────────────────────────────────

def http_poll_listener(engine: RealtimeInference):
    interval = 1.0 / POLL_HZ
    print(f"HTTP polling: {SERVER}/api/v1/sensing/latest @ {POLL_HZ}Hz")
    while True:
        try:
            r = requests.get(f"{SERVER}/api/v1/sensing/latest", timeout=0.5)
            if r.status_code == 200:
                nodes = r.json().get("nodes", [])
                if nodes:
                    engine.push_nodes(nodes)
        except:
            pass
        time.sleep(interval)

# ── Push to server ────────────────────────────────────────────────────────────

def push_prediction_loop(engine: RealtimeInference):
    while True:
        time.sleep(0.1)
        pred  = engine.get_prediction()
        label = pred["label"]
        conf  = pred["confidence"]

        try:
            requests.post(
                f"{SERVER}/api/v1/inference/push",
                json={"posture": label, "confidence": conf, "keypoints": pred["keypoints"]},
                timeout=0.5
            )
        except:
            pass

        if int(time.time() * 10) % 20 == 0:
            fall_state = engine.fall_det.state
            state_icon = "🚨" if fall_state == "fallen" else ("⚠️ " if fall_state == "candidate" else "  ")
            bar = "█" * int(conf * 20)
            print(f"\r{state_icon} {label:12s} {bar:<20} {conf*100:.0f}%", end="", flush=True)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    meta_path = args.model.replace(".pt", "_meta.json")
    engine    = RealtimeInference(args.model, meta_path)

    t1 = threading.Thread(target=http_poll_listener,   args=(engine,), daemon=True)
    t2 = threading.Thread(target=push_prediction_loop, args=(engine,), daemon=True)
    t1.start()
    t2.start()

    print("Realtime inference + fall detection running... Ctrl+C to stop")
    print(f"Fall alert: lying {FALL_CONFIRM_SEC:.0f}s+ → beep + voice")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped")

if __name__ == "__main__":
    main()
