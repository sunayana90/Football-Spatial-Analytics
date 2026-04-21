# ⚽ Football Match Analyzer

A computer vision pipeline that detects players, classifies teams, tracks movement, and generates a **top-view tactical radar** with heatmaps — all from a standard match video.

---

## 🎯 What It Does

- Detects players in every frame using **YOLOv8**
- Classifies players into **Team A or Team B** by jersey color (HSV)
- Tracks each player across frames with a **unique ID**
- Maps player positions from camera view to a **top-down pitch** using homography
- Generates a **rolling heatmap** showing player activity zones
- Outputs a clean **top-view radar video** (no original footage, just the tactical view)

---

## 📽️ Output

| File | Description |
|------|-------------|
| `topview_radar.mp4` | Top-down pitch with player dots, trails, and heatmap overlay |

---

## 🚀 Quick Start (Google Colab)

### Step 1 — Install dependencies
```python
!pip install ultralytics opencv-python-headless numpy scipy
```

### Step 2 — Import libraries
```python
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict, deque
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

YOLO_AVAILABLE = True
SCIPY_AVAILABLE = True
```

### Step 3 — Paste all class definitions
Paste in order: `Config` → `PitchRenderer` → `HomographyManager` → `TeamClassifier` → `HeatmapBuilder` → `SimpleTracker` → `FootballAnalyser`

### Step 4 — Download video & run
```python
import urllib.request

video_url = "https://raw.githubusercontent.com/sunayana90/virtual-ad-overlay-homography-yolo/main/football.mp4"
video_filename = "football.mp4"

urllib.request.urlretrieve(video_url, video_filename)

manual_corners = [
    [100, 50],    # Top-Left
    [1180, 50],   # Top-Right
    [1180, 670],  # Bottom-Right
    [100, 670],   # Bottom-Left
]

analyser = FootballAnalyser(
    video_path=video_filename,
    output_dir="output",
    manual_corners=manual_corners
)
analyser.run()
```

### Step 5 — Download output
```python
from google.colab import files
files.download("output/topview_radar.mp4")
```

---

## 🔧 Configuration

All settings live in the `Config` class. Key parameters you may want to tweak:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `YOLO_MODEL` | `yolov8n.pt` | Model size: `n`=fastest, `s/m/l`=more accurate |
| `CONF_THRESHOLD` | `0.40` | Minimum detection confidence (0–1) |
| `FRAME_SKIP` | `2` | Process every Nth frame (higher = faster) |
| `HEATMAP_SIGMA` | `18` | Heatmap blur radius (higher = smoother) |
| `HEATMAP_ALPHA` | `0.65` | Heatmap overlay transparency (0–1) |
| `TRACK_HISTORY` | `30` | Number of frames to show player trail |

### Team Color Tuning (HSV ranges)

By default the pipeline detects **red** (Team A) and **blue** (Team B) jerseys. To change this, update these values in `Config`:

```python
# Example: switching Team A to yellow
TEAM_A_HSV_LOWER = np.array([20, 80, 80])
TEAM_A_HSV_UPPER = np.array([35, 255, 255])
```

Use [this HSV color picker](https://colorpicker.me/) to find the right range for your team colors.

---

## 📐 Setting Corner Points

The homography transform needs 4 pitch corner coordinates from your video frame. To find them:

```python
import cv2
from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture("football.mp4")
ret, frame = cap.read()
cap.release()

print("Frame size:", frame.shape)  # (height, width, channels)
cv2_imshow(frame)
```

Then update `manual_corners` with the pixel positions of the four pitch corners in order: **Top-Left → Top-Right → Bottom-Right → Bottom-Left**.

---

## 🏗️ Architecture

```
football.mp4
     │
     ▼
┌─────────────┐
│  YOLOv8     │  ← detects people in each frame
└──────┬──────┘
       │ bounding boxes
       ▼
┌─────────────┐
│TeamClassifier│  ← HSV jersey color → Team A / B / unknown
└──────┬──────┘
       │ team labels
       ▼
┌─────────────┐
│SimpleTracker│  ← nearest-center matching → consistent player IDs
└──────┬──────┘
       │ tracked positions
       ▼
┌──────────────────┐
│HomographyManager │  ← camera view → top-down pitch coordinates
└──────┬───────────┘
       │ pitch-space points
       ▼
┌─────────────┐    ┌───────────────┐
│HeatmapBuilder│    │ PitchRenderer │
└──────┬──────┘    └──────┬────────┘
       │                  │
       └────────┬─────────┘
                ▼
        topview_radar.mp4
```

---

## ⚠️ Common Issues

**All players cluster in one corner of the pitch**
→ Your `manual_corners` don't cover the full visible pitch area. Re-check the pixel coordinates.

**No detections / empty video**
→ Lower `CONF_THRESHOLD` (e.g. to `0.25`) or use a larger YOLO model (`yolov8s.pt`).

**Auto corner detection failed**
→ Expected in Colab. Always use `manual_corners` as shown above.

**Wrong team colors**
→ Tune the HSV ranges in `Config` to match your video's jersey colors.

---

## 📦 Requirements

```
ultralytics
opencv-python-headless
numpy
scipy
```

