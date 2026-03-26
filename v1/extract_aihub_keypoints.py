#!/usr/bin/env python3
"""
AI Hub 낙상 이미지 → YOLO 키포인트 추출
외장하드 이미지 → v1/data/camera_dataset/aihub_fall_dataset.json

사용법:
  python3 v1/extract_aihub_keypoints.py
"""

import json, os, glob
from pathlib import Path
import numpy as np
from ultralytics import YOLO

IMAGE_ROOT = "/Volumes/T7 Shield/image/Y"
OUTPUT     = "v1/data/camera_dataset/aihub_fall_dataset.json"
WINDOW     = 30
STRIDE     = 5
FEATURES   = 34  # 17 kp × 2

def extract_keypoints_from_folder(yolo, folder_path):
    """폴더 안 이미지들에서 키포인트 시퀀스 추출"""
    images = sorted(glob.glob(str(folder_path / "*.jpg")) +
                    glob.glob(str(folder_path / "*.JPG")))
    if len(images) < 3:
        return []

    frames = []
    for img_path in images:
        results = yolo(img_path, verbose=False)
        if results and results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
            kp_raw = results[0].keypoints.xyn[0].cpu().numpy()
            valid = np.count_nonzero(kp_raw.sum(axis=1))
            if valid >= 8:
                frames.append(kp_raw.flatten().tolist())
            else:
                frames.append([0.0] * FEATURES)
        else:
            frames.append([0.0] * FEATURES)

    return frames

def frames_to_windows(frames):
    """프레임 시퀀스 → 슬라이딩 윈도우"""
    windows = []
    if len(frames) < WINDOW:
        # 짧으면 패딩
        pad = [[0.0] * FEATURES] * (WINDOW - len(frames))
        frames = pad + frames
        windows.append(frames)
    else:
        for i in range(0, len(frames) - WINDOW + 1, STRIDE):
            windows.append(frames[i:i+WINDOW])
    return windows

def main():
    print("Loading YOLO...")
    yolo = YOLO("yolov8n-pose.pt")

    fall_windows = []
    total_folders = 0
    skipped = 0

    for subfolder in ["FY", "SY"]:
        base = Path(IMAGE_ROOT) / subfolder
        if not base.exists():
            print(f"없음: {base}")
            continue

        folders = sorted(base.iterdir())
        print(f"\n{subfolder}: {len(folders)}개 폴더 처리 중...")

        for i, folder in enumerate(folders):
            if not folder.is_dir():
                continue
            frames = extract_keypoints_from_folder(yolo, folder)
            windows = frames_to_windows(frames)
            fall_windows.extend(windows)
            total_folders += 1

            if (i+1) % 50 == 0:
                print(f"  {i+1}/{len(folders)} 완료, 누적 윈도우: {len(fall_windows)}")

    print(f"\n총 {total_folders}개 폴더 → {len(fall_windows)}개 fall 윈도우")

    os.makedirs("v1/data/camera_dataset", exist_ok=True)
    data = {"data": {"fall": fall_windows, "normal": []}}
    with open(OUTPUT, "w") as f:
        json.dump(data, f)
    print(f"저장: {OUTPUT}")

if __name__ == "__main__":
    main()
