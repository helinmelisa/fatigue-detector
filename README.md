# Fatigue Detector

Real-time driver fatigue detection using OpenCV and MediaPipe. Detects drowsiness and yawning by analyzing facial landmarks from a webcam or video file.

## How it works

- **EAR (Eye Aspect Ratio)** — measures eye openness. Sustained closure triggers a drowsiness alert.
- **MAR (Mouth Aspect Ratio)** — measures mouth openness. Sustained opening triggers a yawn alert.
- **Fatigue Score (0–100)** — combines EAR and MAR into a single score with three levels:
  - `OK` — no fatigue detected
  - `WARNING` — early signs of fatigue
  - `CRITICAL` — high fatigue, immediate alert

## Project structure

```
fatigue_detector/
├── main.py               # Entry point
├── requirements.txt
└── src/
    ├── detector.py       # Main FatigueDetector class
    ├── landmarks.py      # MediaPipe landmark indices
    ├── ear.py            # Eye aspect ratio logic
    ├── mar.py            # Mouth aspect ratio logic
    └── fatigue_score.py  # Fatigue scoring and levels
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Webcam (default camera index 1)
python main.py

# Specific camera index
python main.py --source 0

# Video file
python main.py --source path/to/video.avi
```

Press `q` to quit.

## Dataset

Tested on the [YawDD dataset](https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset) — a yawning detection dataset with dashboard and mirror-mounted camera views.
