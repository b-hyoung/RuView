#!/usr/bin/env python3
"""
Camera Fall Detection - LSTM Training
관절 좌표 시계열 → 낙상/정상/제스처 분류 모델 학습

사용법:
  python3 v1/train_camera_model.py --dataset v1/data/camera_dataset/cam_session_..._dataset.json
"""

import json, os, argparse, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from datetime import datetime

DEVICE    = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_DIR = "v1/data/models"
EPOCHS    = 80
BATCH     = 32
LR        = 3e-4
WINDOW    = 30
FEATURES  = 34   # 17 keypoints × 2 (x, y)
LABELS    = ["fall", "normal"]  # 알파벳 순

class CameraDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path) as f:
            d = json.load(f)

        self.X, self.y = [], []
        for label_idx, label in enumerate(LABELS):
            windows = d["data"].get(label, [])
            for window in windows:
                self.X.append(window)
                self.y.append(label_idx)
            print(f"  {label:8s}: {len(windows)} windows")

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
        print(f"\nTotal: {len(self.X)} windows")

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class FallNet(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(FEATURES, 128, num_layers=2, batch_first=True,
                            dropout=0.3, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

def merge_datasets(paths: list) -> dict:
    merged = {"normal": [], "fall": []}
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        for cat in merged:
            merged[cat].extend(d["data"].get(cat, []))
        print(f"  Loaded: {path}")
    return merged

def train(dataset_paths: list):
    print(f"\n=== Camera Fall Detection Training ===")
    print(f"Device: {DEVICE}")

    if len(dataset_paths) > 1:
        print(f"Merging {len(dataset_paths)} datasets...")
        merged = merge_datasets(dataset_paths)
        # 임시 파일로 저장
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"session": "merged", "counts": {k: len(v) for k,v in merged.items()}, "data": merged}, tmp)
        tmp.close()
        dataset_path = tmp.name
    else:
        dataset_path = dataset_paths[0]

    dataset = CameraDataset(dataset_path)
    if len(dataset) == 0:
        print("No data")
        return

    n_val   = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH)

    print(f"\nTrain: {n_train}  Val: {n_val}")

    model  = FallNet(len(LABELS)).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None

    for epoch in range(1, EPOCHS+1):
        model.train()
        correct, total = 0, 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            loss = loss_fn(model(X), y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (model(X).argmax(1) == y).sum().item()
            total   += len(y)
        sched.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                vc += (model(X).argmax(1) == y).sum().item()
                vt += len(y)
        val_acc = vc / vt * 100

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train={correct/total*100:.1f}%  val={val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val accuracy: {best_val_acc:.1f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODEL_DIR, f"fall_camera_{ts}.pt")
    meta_path  = os.path.join(MODEL_DIR, f"fall_camera_{ts}_meta.json")

    torch.save(best_state, model_path)
    with open(meta_path, "w") as f:
        json.dump({"labels": LABELS, "features": FEATURES, "window": WINDOW,
                   "val_acc": round(best_val_acc, 2), "model_path": model_path}, f, indent=2)

    print(f"Saved: {model_path}")
    print(f"\nNext step:")
    print(f"  python3 v1/camera_inference.py --model {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, action="append", dest="datasets")
    args = parser.parse_args()
    train(args.datasets)
