#!/usr/bin/env python3
"""
Camera Fall Detection - Keypoint Extraction
YOLOv8-Pose로 영상에서 관절 좌표 추출 → 학습 데이터셋 생성

사용법:
  python3 v1/extract_camera_keypoints.py --session cam_session_20260326_120000
  python3 v1/extract_camera_keypoints.py --latest
"""

import cv2, json, os, argparse, numpy as np
from ultralytics import YOLO

SESSIONS_DIR = "v1/data/camera_sessions"
DATASET_DIR  = "v1/data/camera_dataset"
WINDOW       = 30   # 프레임 수
STRIDE       = 5
N_KP         = 17   # YOLO COCO keypoints
FEATURES     = N_KP * 2  # x, y per keypoint = 34

def get_latest_session():
    sessions = sorted(os.listdir(SESSIONS_DIR))
    for s in reversed(sessions):
        if os.path.exists(os.path.join(SESSIONS_DIR, s, "session_log.json")):
            return s
    return None

def extract_keypoints_from_video(video_path: str, model: YOLO) -> list:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints.xyn  # normalized [0,1]
            if len(kps) > 0:
                # 첫 번째 사람만 사용
                kp = kps[0].cpu().numpy()  # (17, 2)
                flat = kp.flatten().tolist()  # 34개
                frames.append({"keypoints": flat, "detected": True})
            else:
                frames.append({"keypoints": [0.0] * FEATURES, "detected": False})
        else:
            frames.append({"keypoints": [0.0] * FEATURES, "detected": False})
    cap.release()
    detected = sum(1 for f in frames if f["detected"])
    print(f"    {len(frames)} frames, {detected} detected ({detected/max(len(frames),1)*100:.0f}%)")
    return frames

def process_session(session_id: str):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    log_path    = os.path.join(session_dir, "session_log.json")

    if not os.path.exists(log_path):
        print(f"Session log not found: {log_path}")
        return

    with open(log_path) as f:
        log = json.load(f)

    print("Loading YOLOv8-Pose model...")
    model = YOLO("yolov8n-pose.pt")  # auto-download ~6MB

    os.makedirs(DATASET_DIR, exist_ok=True)
    dataset = {"normal": [], "fall": []}

    for rec in log["recordings"]:
        label    = rec["label"]
        category = rec["category"]  # fall / normal / gesture

        video_files = [f for f in os.listdir(session_dir)
                       if f.startswith(label) and f.endswith(".mp4")]
        if not video_files:
            print(f"  [{label}] No video file, skipping")
            continue

        video_path = os.path.join(session_dir, video_files[0])
        print(f"  [{category}] {label} ...")

        frames = extract_keypoints_from_video(video_path, model)
        kp_list = [f["keypoints"] for f in frames if f["detected"]]

        if len(kp_list) < WINDOW:
            print(f"    Too few frames ({len(kp_list)}), skipping")
            continue

        # 슬라이딩 윈도우
        windows = 0
        for i in range(0, len(kp_list) - WINDOW, STRIDE):
            window = kp_list[i:i+WINDOW]  # (30, 34)
            dataset[category].append(window)
            windows += 1
        print(f"    → {windows} windows")

    # 저장
    out_path = os.path.join(DATASET_DIR, f"{session_id}_dataset.json")
    counts = {k: len(v) for k, v in dataset.items()}
    with open(out_path, "w") as f:
        json.dump({"session": session_id, "counts": counts, "data": dataset}, f)

    print(f"\n=== Done ===")
    for cat, count in counts.items():
        print(f"  {cat:8s}: {count:4d} windows")
    print(f"\nSaved: {out_path}")
    print(f"Next step:")
    print(f"  python3 v1/train_camera_model.py --dataset {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=str)
    parser.add_argument("--latest", action="store_true")
    args = parser.parse_args()

    if args.latest or not args.session:
        session_id = get_latest_session()
        if not session_id:
            print("No session found. Run collect_camera_data.py first.")
            return
        print(f"Using latest session: {session_id}")
    else:
        session_id = args.session

    process_session(session_id)

if __name__ == "__main__":
    main()
