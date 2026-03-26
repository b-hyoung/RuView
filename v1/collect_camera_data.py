#!/usr/bin/env python3
"""
Camera Fall Detection - Data Collection
맥캠으로 낙상/정상/제스처 영상 수집

사용법:
  python3 v1/collect_camera_data.py
"""

import cv2, os, time, json
from datetime import datetime

OUTPUT_DIR = "v1/data/camera_sessions"

AUTO_SEQUENCE = [
    ("normal_walk",    "카메라 앞에서 자연스럽게 걷기",        25),
    ("normal_sit",     "의자에 앉았다 일어나기 반복",          25),
    ("normal_lie",     "천천히 바닥에 눕기 (의도적으로)",       20),
    ("fall_forward",   "앞으로 쓰러지기 x5 (매트 위에서!)",    30),
    ("fall_sideways",  "옆으로 쓰러지기 x5 (매트 위에서!)",    30),
    ("fall_kneel",     "무릎 꺾으며 쓰러지기 x5",             30),
]

# fall_* → "fall" / normal_* → "normal"
LABEL_MAP = {
    "normal_walk": "normal", "normal_sit": "normal", "normal_lie": "normal",
    "fall_forward": "fall",  "fall_sideways": "fall", "fall_kneel": "fall",
}

def draw_overlay(frame, state):
    h, w = frame.shape[:2]
    label    = state["label"]
    desc     = state["desc"]
    elapsed  = state["elapsed"]
    step     = state["step"]
    total    = state["total"]
    rec      = state["recording"]
    category = LABEL_MAP.get(label, "?")

    color = {"fall": (0,0,220), "normal": (0,200,0), "gesture": (200,150,0)}.get(category, (200,200,200))

    if rec:
        cv2.rectangle(frame, (0,0), (w-1,h-1), color, 6)
        cv2.circle(frame, (28,28), 10, color, -1)
        cv2.putText(frame, f"REC  {elapsed:.0f}s", (48,38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.rectangle(frame, (0,h-80), (w,h), (0,0,0), -1)
        cv2.putText(frame, f"[{category.upper()}] {label}", (10,h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, desc, (10,h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
    else:
        cv2.rectangle(frame, (0,0), (w,90), (0,0,0), -1)
        cv2.putText(frame, f"[{step}/{total}] {label}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, desc, (10,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.rectangle(frame, (0,h-45), (w,h), (0,0,0), -1)
        cv2.putText(frame, "SPACE: Start  S: Skip  Q: Quit", (10,h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,220,255), 1)
    return frame

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    session_id  = datetime.now().strftime("cam_session_%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera permission needed: System Settings → Privacy → Camera → Terminal")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    log = []
    total = len(AUTO_SEQUENCE)

    for step_idx, (label, desc, suggested_sec) in enumerate(AUTO_SEQUENCE):
        print(f"\n[{step_idx+1}/{total}] {label} — {desc}")
        print(f"  Press SPACE to start (suggested {suggested_sec}s)")

        state = dict(recording=False, label=label, desc=desc,
                     elapsed=0.0, step=step_idx+1, total=total)
        rec_start    = 0.0
        video_writer = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            state["elapsed"] = time.time() - rec_start if state["recording"] else 0.0
            cv2.imshow("Camera Data Collector", draw_overlay(frame.copy(), state))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):
                if state["recording"] and video_writer:
                    video_writer.release()
                cap.release()
                cv2.destroyAllWindows()
                _save_log(session_dir, session_id, log)
                return

            if key in (ord('s'), ord('S')) and not state["recording"]:
                print(f"  Skipped: {label}")
                break

            if key == ord(' '):
                if not state["recording"]:
                    ts      = datetime.now().strftime("%H%M%S")
                    vpath   = os.path.join(session_dir, f"{label}_{ts}.mp4")
                    video_writer = cv2.VideoWriter(vpath, fourcc, 30, (640,480))
                    rec_start = time.time()
                    state["recording"] = True
                    print("  Recording... (SPACE to stop)")
                else:
                    duration = time.time() - rec_start
                    state["recording"] = False
                    video_writer.release(); video_writer = None
                    log.append({"label": label, "category": LABEL_MAP[label],
                                "file": os.path.basename(vpath), "duration_sec": round(duration,1)})
                    print(f"  Done: {duration:.0f}s")
                    break

            # auto-stop
            if state["recording"] and (time.time() - rec_start) >= suggested_sec:
                duration = time.time() - rec_start
                state["recording"] = False
                video_writer.release(); video_writer = None
                log.append({"label": label, "category": LABEL_MAP[label],
                            "file": os.path.basename(vpath), "duration_sec": round(duration,1)})
                print(f"  Auto done: {duration:.0f}s")
                time.sleep(0.5)
                break

            if state["recording"] and video_writer:
                video_writer.write(frame)

    cap.release()
    cv2.destroyAllWindows()
    _save_log(session_dir, session_id, log)

def _save_log(session_dir, session_id, log):
    if not log:
        return
    log_path = os.path.join(session_dir, "session_log.json")
    with open(log_path, "w") as f:
        json.dump({"session_id": session_id, "recordings": log}, f, indent=2)
    print(f"\n=== Done ===")
    for e in log:
        print(f"  [{e['category']:7s}] {e['label']:15s} {e['duration_sec']:4.0f}s")
    print(f"\nNext step:")
    print(f"  python3 v1/extract_camera_keypoints.py --session {session_id}")

if __name__ == "__main__":
    main()
