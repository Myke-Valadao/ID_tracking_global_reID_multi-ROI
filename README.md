# Persistent ID Tracking + Global ReID with Multi-ROI (Ultralytics YOLO)

This repository contains a single Python script that performs **persistent multi-object tracking** with **global Re-Identification (ReID)** and **multi-polygon Regions of Interest (ROIs)**. It integrates:

- **Ultralytics YOLO** for detection + built-in trackers (ByteTrack / BoT-SORT / OC-SORT / StrongSORT) for **local IDs**.
- **Global ReID** via **torchvision** backbones (ResNet / MobileNetV3) to keep a gallery of embeddings that persists across frames.
- **ROI gating**: **tracking and ReID are activated only when a detection intersects your ROIs** (one or more polygons).
- **Profiling per frame** (CSV): YOLO times (pre/inf/post), embedding time, FPS (instant/avg), and CUDA memory.
- **Frame-level ID uniqueness rule**: **two different boxes in the same frame will never share the same Global ID (GID)**.

> Tip: You can **draw ROIs interactively** at the first frame or **load them from a JSON file**. ROIs can also be saved for reuse.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Command-Line Usage](#command-line-usage)
4. [Key Features & Concepts](#key-features--concepts)
5. [ROI Modes & JSON Format](#roi-modes--json-format)
6. [Outputs](#outputs)
7. [Examples](#examples)
8. [Performance Tips](#performance-tips)
9. [Troubleshooting](#troubleshooting)
10. [FAQ](#faq)
11. [Reproducibility Notes](#reproducibility-notes)
12. [License](#license)

---

## Requirements

- **Python** ≥ 3.9 (tested with 3.10–3.12)
- **PyTorch** + **torchvision** (with CUDA if you want GPU)
- **ultralytics** (YOLO)
- **OpenCV** (`opencv-python`)
- **NumPy**

> For GPU acceleration: install PyTorch with CUDA matching your driver. See the PyTorch website for the correct command.

---

## Installation

```bash
# 1) Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # example for CUDA 12.1
# or for CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install ultralytics opencv-python numpy

# 3) (Optional) Verify YOLO works
yolo help
```

> **Models/Trackers:** Ensure that your `--weights` file exists (e.g., `yolo11n.pt`) and your `--tracker` YAML is available
> (Ultralytics includes sample configs like `bytetrack.yaml`, `botsort.yaml`, `ocsort.yaml`, `strongsort.yaml`).

---

## Command-Line Usage

```bash
python id_tracking_global_reid_multi-roi.py \
  --sources "<0, path/to/video|path/to/image_or_dir>" \
  [--weights yolo11n.pt] \
  [--tracker bytetrack.yaml] \
  [--device cuda:0|cpu] \
  [--imgsz 640] \
  [--class-filter "person,cell phone"] \
  [--match-thresh 0.6] \
  [--gallery-ttl 600] \
  [--emb-backbone resnet50|resnet18|mobilenetv3_small|mobilenetv3_large] \
  [--save output.mp4] \
  [--log-csv run.csv] \
  [--print-every 30] \
  [--roi-mode interactive|file|none] \
  [--roi-file rois.json] \
  [--roi-save rois_saved.json] \
  [--roi-min-intersection 0.3]
```

### Arguments

- `--sources` **(required)**: Camera index (e.g., `0, 1`) or a path to video/image (a directory is also accepted by Ultralytics).
- `--weights`: YOLO weights; default `yolo11n.pt`.
- `--tracker`: Tracker config YAML (`bytetrack.yaml` by default). Other options: `botsort.yaml`, `ocsort.yaml`, `strongsort.yaml`.
- `--device`: `cuda:0` or `cpu`. If omitted, auto-selects GPU if available, otherwise CPU.
- `--imgsz`: Inference size (default `640`). Larger can increase accuracy at cost of speed.
- `--class-filter`: Comma-separated class names to keep (e.g., `"person,car"`). Names must match the model’s class names.
- `--match-thresh`: Cosine similarity threshold for **Global ReID** match (default `0.6`). Higher → stricter matches.
- `--gallery-ttl`: Time-to-live in seconds for gallery entries (default `600`). Old entries are pruned when unseen for longer than TTL.
- `--emb-backbone`: Embedding backbone for ReID. Choices: `resnet50` (default), `resnet18`, `mobilenetv3_small`, `mobilenetv3_large`.
- `--save`: Output video path (`.mp4`). If set, annotated frames are written.
- `--log-csv`: Path to profiling CSV. If set, per-frame metrics are saved (see [Outputs](#outputs)).
- `--print-every`: Print periodic stats every N frames (default `30`). Set `0` to disable.
- **ROI options**:
  - `--roi-mode`: `interactive` (draw polygons at the first frame), `file` (load from JSON), or `none` (disable ROI gating).
  - `--roi-file`: Path to JSON with polygons (required if `--roi-mode file`).
  - `--roi-save`: If set, saves polygons drawn interactively to this JSON file.
  - `--roi-min-intersection`: Minimum fraction of bbox area inside the ROI to **enable tracking/ReID** for that bbox (default `0.3`).

### Controls

- **Interactive ROI mode window keys**:
  - **Mouse Left Click**: add a point.
  - **ENTER**: close current polygon (needs ≥ 3 points).
  - **n**: start a new polygon (closes current one if ≥ 3 points).
  - **u**: undo last point (or last polygon if current is empty).
  - **c**: clear all.
  - **q**: finish ROI editing.
- **Main viewer**: press **q** to quit the run.

---

## Key Features & Concepts

### Local IDs vs Global IDs (GID)
- **Local IDs** come from the tracker (e.g., ByteTrack). They are stable while the tracker keeps the trajectory.
- **Global IDs (GID)** come from our **gallery-based ReID**. The script extracts an embedding for each ROI-qualified detection, then:
  1. **Matches** it to the closest embedding in the class-specific gallery using cosine similarity.
  2. If similarity ≥ `--match-thresh`, it **reuses** that GID and **updates** its embedding via EMA.
  3. Otherwise, it **creates a new GID**.
- **Uniqueness rule per frame**: if two detections would get the same GID in the **same frame**, the second one is forced to a **new GID**.

### ROI Gating
- Tracking/ReID only **activate** when the detection **intersects** the ROI(s) with area overlap ≥ `--roi-min-intersection`.
- Detections **outside** ROI(s) are drawn in gray and **do not** consume ReID/compute.

### Embedding Backbones
- **resnet50** (default): strongest embeddings, heavier.
- **resnet18**: lighter, faster.
- **mobilenetv3_small/large**: mobile-friendly options.
- All are normalized and updated with EMA on each match.

---

## ROI Modes & JSON Format

### Modes
- `interactive`: draw polygons at first frame (recommended for quick setup).
- `file`: load polygons from disk (headless-safe / automation).
- `none`: disable ROI gating (track everything).

### JSON Format
`--roi-file` / `--roi-save` contains a list of polygons; each polygon is a list of `[x, y]` integer points.

Example (`rois.json`):
```json
[
  [[120, 300], [400, 320], [420, 540], [100, 520]],
  [[800, 200], [1000, 220], [980, 480], [780, 460]]
]
```

---

## Outputs

### 1) Annotated Video (`--save output.mp4`)
- Saves the on-screen visualization: bounding boxes, labels (class + GID when inside ROI), FPS HUD, and ROI overlays.

### 2) Profiling CSV (`--log-csv run.csv`)
Columns:
- `frame_idx`: frame index.
- `n_det`: total detections before ROI gating.
- `n_det_roi`: detections that passed ROI intersection threshold (i.e., tracking/ReID ran).
- `yolo_pre_ms`, `yolo_inf_ms`, `yolo_post_ms`: YOLO preprocess/inference/postprocess times (ms) from Ultralytics.
- `emb_total_ms`: total time (ms) spent computing embeddings for the frame.
- `emb_avg_ms`: average embedding time per ROI-qualified detection in the frame.
- `fps_inst`, `fps_avg`: instantaneous and moving-average FPS.
- `cuda_mem_mb`: allocated CUDA memory (MB).

> You can analyze this CSV to understand bottlenecks and tune parameters.

---

## Examples

### Webcam (GPU), draw ROIs, save video + CSV
```bash
python id_tracking_global_reid_multi-roi.py \
  --sources "0, 1" \
  --weights yolo11n.pt \
  --tracker bytetrack.yaml \
  --device cuda:0 \
  --imgsz 640 \
  --roi-mode interactive \
  --roi-save rois_webcam.json \
  --roi-min-intersection 0.3 \
  --class-filter "person" \
  --emb-backbone resnet50 \
  --save outputs/webcam_annotated.mp4 \
  --log-csv outputs/webcam_run.csv \
  --print-every 60
```

### Video file (CPU), load ROIs from JSON
```bash
python id_tracking_global_reid_multi-roi.py \
  --sources "0, /path/to/video.mp4" \
  --device cpu \
  --roi-mode file \
  --roi-file rois.json \
  --class-filter "person,cell phone" \
  --save outputs/video_annotated.mp4
```

### Headless server (no GUI): force ROI from file
```bash
python id_tracking_global_reid_multi-roi.py \
  --sources "0, /data/cam_feed.mp4" \
  --device cuda:0 \
  --roi-mode file \
  --roi-file /data/rois.json \
  --log-csv /logs/run.csv \
  --save /outputs/out.mp4
```

### Tighter ReID matching + shorter memory
```bash
python id_tracking_global_reid_multi-roi.py \
  --sources "0, 1" \
  --match-thresh 0.75 \
  --gallery-ttl 180
```

### Lighter ReID backbone for speed
```bash
python id_tracking_global_reid_multi-roi.py \
  --sources "0, 1" \
  --emb-backbone mobilenetv3_small
```

---

## Performance Tips

- Prefer **GPU** (`--device cuda:0`) for real-time performance.
- Use a **smaller model** (`yolo11n.pt`) and **lower `--imgsz`** (e.g., 480–640) to boost FPS.
- Restrict to **target classes** with `--class-filter` to reduce embedding calls.
- Increase `--roi-min-intersection` (e.g., `0.5`) to be stricter about when to run ReID.
- Use a **lighter embedding backbone** (e.g., `resnet18` or `mobilenetv3_small`) if embedding time dominates.
- Write outputs to a **fast disk** (SSD) to avoid IO stalls when using `--save`.
- If your camera is high-FPS, consider an external **frame skip** (not built-in) to reduce load.

---

## Troubleshooting

### OpenCV GUI errors (headless/SSH/Wayland)
If you see an error like “Could not create OpenCV GUI window” during `--roi-mode interactive`, you are likely in a headless session or lack a proper display.
**Solutions:**
1. Run locally with a desktop session (check `$DISPLAY`), or
2. Switch to `--roi-mode file` using a prepared JSON (see [JSON Format](#json-format)).

### No tracker IDs appear
- Ensure your `--tracker` YAML exists and is supported by Ultralytics.
- Some trackers need additional dependencies. Check Ultralytics docs if required.

### Class names don’t match
- `--class-filter` uses **model class names**. Print `model.names` in code or check your YOLO model’s metadata.

### CUDA memory shows 0
- Happens on CPU runs or if the GPU is not visible. Verify `torch.cuda.is_available()` and drivers.

---

## FAQ

**Q: How are Global IDs (GIDs) maintained?**  
A: Per class, we keep a gallery `{gid -> normalized embedding, last_seen}`. Each ROI-qualified detection gets an embedding, matched via cosine similarity. If `sim ≥ --match-thresh`, we reuse that `gid` and update its embedding via **EMA** (`0.9*old + 0.1*new`); otherwise a new `gid` is created. Old entries are pruned after `--gallery-ttl` seconds.

**Q: What happens if two detections match the same GID in the same frame?**  
A: The script enforces **uniqueness per frame**: a second detection will be forced to a **new GID** to avoid collisions.

**Q: Can I disable ROI gating?**  
A: Yes, set `--roi-mode none`. All detections will run through tracking/ReID.

**Q: Can I run on images or a folder of images?**  
A: Ultralytics supports images/folders. The script uses `model.track(..., stream=True)`, so typical file/folder inputs should work.
(For single images, tracking is trivial; ReID will still produce GIDs.)

**Q: How do I choose a tracker?**  
A: Start with **ByteTrack** for simplicity and speed. If you need appearance cues in the tracker itself, try **StrongSORT**.

---

## Reproducibility Notes

- Fix random seeds where applicable (Ultralytics uses internal seeds for training; for inference this is less critical).
- Keep versions pinned in `requirements.txt` or your `pip freeze` to ensure stable FPS and behavior across environments.
- Record your **weights**, **tracker YAML**, **imgsz**, **ROI JSON**, and **CLI args** alongside results.

---

## License

This README documents a script that depends on third-party libraries. Please check each dependency’s license (e.g., Ultralytics, PyTorch, Torchvision, OpenCV).  

---

### Attributions

- Ultralytics YOLO (detection + trackers)
- Torchvision (embedding backbones)

---

## Short Reference (Cheat Sheet)

- Run webcam on GPU, draw ROIs, save:  
  `python id_tracking_global_reid_multi-roi.py --sources "0, 1" --device cuda:0 --roi-mode interactive --save out.mp4 --log-csv run.csv`

- Load ROIs from JSON, restrict classes, CPU:  
  `python id_tracking_global_reid_multi-roi.py --sources "0, video.mp4" --device cpu --roi-mode file --roi-file rois.json --class-filter "person"`

- Faster ReID:  
  `--emb-backbone resnet18` or `--emb-backbone mobilenetv3_small`

- Stricter identity matches:  
  `--match-thresh 0.75`

- Prune old identities faster:  
  `--gallery-ttl 180`
