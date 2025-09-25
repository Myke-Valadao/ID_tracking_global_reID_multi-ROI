#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Source Persistent ID Tracking + Global ReID (profiling) + Múltiplas ROIs + Mosaico Único
- Várias fontes simultâneas: webcams, vídeos, RTSP, etc. via --sources "0,1,/path,rtsp://..."
- Uma thread por fonte: YOLO (Ultralytics) + tracker (ByteTrack/BOT/OC/StrongSORT) para IDs locais.
- ReID global compartilhado entre câmeras (galeria única) via embeddings (torchvision).
- Profiling por frame (CSV): tempos YOLO, tempo de embedding, FPS, memória CUDA.
- Unicidade por frame -> dois bboxes no mesmo frame nunca compartilham o mesmo GID.
- Ativa tracking/ReID apenas se o bbox INTERSECTA as ROIs (múltiplos polígonos por câmera).
- ROIs por câmera: desenho interativo no 1º frame de cada câmera OU carregadas de JSON (um por câmera).
- Mosaico único (janela "MultiCam Mosaic") com tiles de tamanho unificado (--tile-w/--tile-h).
- Textos e HUD proporcionais ao tamanho do frame original; "Cam X" no canto inferior esquerdo de cada tile.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path
from collections import defaultdict, deque
from threading import Thread, Lock, Event
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO


# =========================
# UI helpers (texto proporcional)
# =========================
def _text_style_for(img, base_factor=0.9):
    """
    Calcula escala e espessura de fonte proporcionais ao tamanho da imagem.
    base_factor: ajuste fino (1.0 = neutro). Use 0.8~1.2 para calibrar.
    """
    h, w = img.shape[:2]
    ref = min(h, w)
    scale = np.clip((ref / 720.0) * base_factor, 0.5, 2.2)
    thickness = max(1, int(round(scale * 2)))
    return float(scale), int(thickness)

def draw_text_proportional(
    img, text, anchor="bl", margin=8, base_factor=0.9,
    color_fg=(255, 255, 255), color_bg=(0, 0, 0)
):
    """
    Desenha texto com sombra no img, com posição por âncora:
    anchor: 'tl' top-left, 'tr' top-right, 'bl' bottom-left, 'br' bottom-right, 'tc' top-center.
    """
    scale, thick = _text_style_for(img, base_factor)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    h, w = img.shape[:2]

    if anchor == "tl":
        x = margin; y = margin + th
    elif anchor == "tr":
        x = w - margin - tw; y = margin + th
    elif anchor == "bl":
        x = margin; y = h - margin
    elif anchor == "br":
        x = w - margin - tw; y = h - margin
    elif anchor == "tc":
        x = (w - tw) // 2; y = margin + th
    else:
        x = margin; y = margin + th

    # sombra
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color_bg, thick + 2, cv2.LINE_AA)
    # frente
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color_fg, thick, cv2.LINE_AA)

def draw_label(img, box, text):
    """
    Rótulo de bbox com escala proporcional ao frame original.
    Centraliza horizontalmente no topo do bbox.
    """
    x1, y1, x2, y2 = box
    scale, thick = _text_style_for(img, base_factor=0.9)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cx = int((x1 + x2) / 2)
    y = max(0, int(y1) - int(6 * scale))
    x = int(cx - tw // 2)
    # sombra
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    # frente
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


# =========================
# Embedding backbones (torchvision)
# =========================
def _default_mean_std():
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def build_embedding_backbone(name: str, device: str):
    import torch.nn as nn
    name = name.lower().strip()
    mean, std = _default_mean_std()

    if name == "resnet50":
        from torchvision.models import resnet50
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            try:
                meta = weights.meta or {}
                mean = meta.get("mean", mean); std = meta.get("std", std)
            except Exception:
                pass
        except Exception:
            model = resnet50(pretrained=True)
        feat_dim = 2048
        feature_model = nn.Sequential(*list(model.children())[:-1])
        input_size = 224

    elif name == "resnet18":
        from torchvision.models import resnet18
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            try:
                meta = weights.meta or {}
                mean = meta.get("mean", mean); std = meta.get("std", std)
            except Exception:
                pass
        except Exception:
            model = resnet18(pretrained=True)
        feat_dim = 512
        feature_model = nn.Sequential(*list(model.children())[:-1])
        input_size = 224

    elif name in ("mobilenetv3_small", "mobilenetv3-small", "mnetv3s"):
        from torchvision.models import mobilenet_v3_small
        try:
            from torchvision.models import MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT
            model = mobilenet_v3_small(weights=weights)
            try:
                meta = weights.meta or {}
                mean = meta.get("mean", mean); std = meta.get("std", std)
            except Exception:
                pass
        except Exception:
            model = mobilenet_v3_small(pretrained=True)
        feature_model = nn.Sequential(
            model.features,
            # "neck" simples para vetor fixo
            nn.Conv2d(576, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960), nn.Hardswish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(960, 1280), nn.Hardswish()
        )
        feat_dim = 1280
        input_size = 224

    elif name in ("mobilenetv3_large", "mobilenetv3-large", "mnetv3l"):
        from torchvision.models import mobilenet_v3_large
        try:
            from torchvision.models import MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.DEFAULT
            model = mobilenet_v3_large(weights=weights)
            try:
                meta = weights.meta or {}
                mean = meta.get("mean", mean); std = meta.get("std", std)
            except Exception:
                pass
        except Exception:
            model = mobilenet_v3_large(pretrained=True)
        feature_model = nn.Sequential(
            model.features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        feat_dim = 1280
        input_size = 224

    else:
        raise ValueError(f"Backbone não suportado: {name}")

    feature_model = feature_model.to(device).eval()
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return feature_model, transform, feat_dim, name


class EmbeddingExtractor:
    def __init__(self, backbone: str, device: str):
        self.device = device
        self.feature_model, self.transform, self.feat_dim, self.name = build_embedding_backbone(backbone, device)

    @torch.no_grad()
    def extract(self, crop_bgr: np.ndarray):
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tens = self.transform(img).unsqueeze(0).to(self.device)
        feat = self.feature_model(tens)
        if feat.ndim == 4:
            feat = torch.flatten(feat, 1)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy()[0]


# =========================
# Global ReID Gallery (compartilhada)
# =========================
class GlobalReIDGallery:
    def __init__(self, match_thresh=0.6, ttl_seconds=600.0, ema=0.9):
        self.match_thresh = match_thresh
        self.ttl = ttl_seconds
        self.ema = ema
        self.next_gid = 1
        self.gallery = defaultdict(dict)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None: return -1.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def _prune(self, now: float):
        for cls, bank in self.gallery.items():
            to_del = [gid for gid, rec in bank.items() if now - rec["last_seen"] > self.ttl]
            for gid in to_del: del bank[gid]

    def create_new(self, cls_name: str, emb: np.ndarray, now: float):
        gid = self.next_gid; self.next_gid += 1
        self.gallery[cls_name][gid] = {
            "emb": emb / (np.linalg.norm(emb) + 1e-12),
            "last_seen": now,
            "label": cls_name,
        }
        return gid

    def match_or_create(self, cls_name: str, emb: np.ndarray, now: float):
        self._prune(now)
        bank = self.gallery[cls_name]
        best_gid, best_sim = None, -1.0
        for gid, rec in bank.items():
            sim = self._cosine_sim(emb, rec["emb"])
            if sim > best_sim: best_sim, best_gid = sim, gid
        if best_gid is not None and best_sim >= self.match_thresh:
            rec = bank[best_gid]
            rec["emb"] = self.ema * rec["emb"] + (1 - self.ema) * emb
            rec["emb"] = rec["emb"] / (np.linalg.norm(rec["emb"]) + 1e-12)
            rec["last_seen"] = now
            return best_gid, False
        return self.create_new(cls_name, emb, now), True


# =========================
# ROI helpers
# =========================
def draw_polygons(img, polygons, color=(20, 180, 255), thickness=2):
    """
    Desenha uma lista de polígonos sobre a imagem.
    polygons: [[(x1,y1),(x2,y2),...], ...]
    """
    if not polygons:
        return
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32)
        if len(pts) >= 3:
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            for p in pts:
                cv2.circle(img, tuple(p), 3, color, -1)

def build_roi_mask(polygons, shape_hw):
    """Cria uma máscara binária (uint8) com 255 dentro de qualquer polígono."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    polys = [np.array(p, dtype=np.int32) for p in polygons if len(p) >= 3]
    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask

def bbox_roi_overlap_ratio(bbox_xyxy, roi_mask):
    """Retorna fração de área do bbox que está DENTRO das ROIs."""
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = roi_mask.shape[:2]
    x1 = max(0, min(w - 1, x1)); x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1)); y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1: return 0.0
    sub = roi_mask[y1:y2, x1:x2]
    area_bbox = (x2 - x1) * (y2 - y1)
    area_in = int((sub > 0).sum())
    return float(area_in) / float(area_bbox + 1e-9)


class ROIEditor:
    """Editor simples de polígonos no primeiro frame."""
    def __init__(self, frame, window_suffix=""):
        self.orig = frame.copy()
        self.img = frame.copy()
        self.polygons = []
        self.current = []
        self.window = f"Desenhe ROIs{window_suffix} (ENTER fecha, n novo, u desfaz, c limpar, q sair)"

        try:
            try:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            except Exception:
                cv2.namedWindow(self.window)
            _test = "_cv2_gui_test_"
            try:
                cv2.namedWindow(_test); cv2.destroyWindow(_test)
            except Exception:
                pass
            cv2.setMouseCallback(self.window, self.on_mouse)
        except cv2.error as e:
            msg = (
                "[ROI] Não foi possível criar janela de GUI do OpenCV.\n"
                "Causas: sessão headless/SSH sem X, DISPLAY ausente, Wayland sem plugin.\n"
                "Soluções:\n"
                "  1) Rode local com GUI habilitada (verifique $DISPLAY).\n"
                "  2) Use --roi-mode file com um JSON de polígonos.\n"
                "Exemplo JSON: [[ [x1,y1], [x2,y2], ... ], [ ... ]]\n"
                f"Erro original: {e}"
            )
            raise RuntimeError(msg)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y)); self.redraw()

    def redraw(self):
        self.img = self.orig.copy()
        draw_polygons(self.img, self.polygons, color=(20, 180, 255))
        if len(self.current) >= 1:
            pts = np.array(self.current, dtype=np.int32)
            cv2.polylines(self.img, [pts], isClosed=False, color=(0, 255, 255), thickness=2)
            for p in self.current:
                cv2.circle(self.img, p, 3, (0, 255, 255), -1)

    def loop(self):
        self.redraw()
        print("[ROI] clique = ponto | ENTER = fecha polígono | n = novo | u = desfaz | c = limpar | q = sair")
        while True:
            cv2.imshow(self.window, self.img)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):  # ENTER
                if len(self.current) >= 3:
                    self.polygons.append(self.current.copy())
                    self.current.clear(); self.redraw()
                    print(f"[ROI] Polígono adicionado. Total: {len(self.polygons)}.")
                else:
                    print("[ROI] Polígono precisa de pelo menos 3 pontos.")
            elif key == ord('n'):
                if len(self.current) >= 3:
                    self.polygons.append(self.current.copy())
                self.current.clear(); self.redraw()
            elif key == ord('u'):
                if self.current:
                    self.current.pop(); self.redraw()
                elif self.polygons:
                    self.polygons.pop(); self.redraw()
            elif key == ord('c'):
                self.polygons.clear(); self.current.clear(); self.redraw()
            elif key == ord('q'):
                break

        cv2.destroyWindow(self.window)
        if len(self.current) >= 3:
            self.polygons.append(self.current.copy()); self.current.clear()
        return self.polygons


# =========================
# Funções de mosaico
# =========================
def safe_resize(img: Optional[np.ndarray], w: int, h: int) -> np.ndarray:
    if img is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

def build_mosaic(frames: List[Optional[np.ndarray]], cols: int, tile_w: int, tile_h: int, labels: List[str]) -> np.ndarray:
    n = len(frames)
    cols = max(1, cols)
    rows = math.ceil(n / cols)
    grid = []
    k = 0
    for _ in range(rows):
        row_imgs = []
        for _ in range(cols):
            if k < n:
                tile = safe_resize(frames[k], tile_w, tile_h)
                # Rótulo "Cam X" no canto inferior esquerdo do tile
                tmp = tile.copy()
                draw_text_proportional(tmp, labels[k], anchor="bl", margin=8, base_factor=0.95)
                row_imgs.append(tmp)
            else:
                row_imgs.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
            k += 1
        grid.append(np.hstack(row_imgs))
    mosaic = np.vstack(grid)
    return mosaic


# =========================
# Worker por câmera
# =========================
def process_stream(
    idx: int,
    source: str,
    args: argparse.Namespace,
    gallery: GlobalReIDGallery,
    gallery_lock: Lock,
    roi_polygons: List[List[Tuple[int,int]]],
    shared_frames: List[Optional[np.ndarray]],
    shared_lock: Lock,
    stop_event: Event,
):
    """
    Processa uma fonte única (thread separada).
    """
    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)  # instância por thread
    names = model.names

    # filtro de classes
    class_set = None
    if args.class_filter:
        wanted = {c.strip() for c in args.class_filter.split(",")}
        class_set = {i for i, n in names.items() if n in wanted}

    emb = EmbeddingExtractor(args.emb_backbone, device)
    local2global = {}

    # Saídas por câmera
    save_path = None
    if args.save:
        save_path = args.save.replace("{idx}", str(idx))
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    csv_writer = None
    csv_file = None
    if args.log_csv:
        csv_path = args.log_csv.replace("{idx}", str(idx))
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_idx","n_det","n_det_roi","yolo_pre_ms","yolo_inf_ms","yolo_post_ms",
                             "emb_total_ms","emb_avg_ms","fps_inst","fps_avg","cuda_mem_mb"])

    # Stream do YOLO com tracker
    results = model.track(
        source=source, imgsz=args.imgsz, tracker=args.tracker,
        conf=0.25, iou=0.45, device=device, stream=True, persist=True, verbose=False
    )

    # ROI (máscara construída do primeiro frame útil)
    roi_mask = None
    roi_ready = (args.roi_mode == "none")
    _roi_polys = roi_polygons  # já definido na preparação em main()

    writer = None
    print(f"[Cam {idx}] Embedding backbone: {emb.name}")
    t_prev = time.perf_counter()
    fps_window = deque(maxlen=60)
    frame_idx = 0

    try:
        for res in results:
            if stop_event.is_set():
                break

            frame = getattr(res, "orig_img", None)
            if frame is None:
                continue
            frame = frame.copy()
            h, w = frame.shape[:2]

            if writer is None and save_path:
                writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

            # Inicialização de ROI (se ainda não)
            if not roi_ready:
                if _roi_polys:
                    roi_mask = build_roi_mask(_roi_polys, (h, w))
                    roi_ready = True
                else:
                    roi_ready = True  # desabilita

            # Desenha ROIs
            if args.roi_mode != "none" and _roi_polys:
                draw_polygons(frame, _roi_polys, color=(20, 180, 255))

            spd = getattr(res, "speed", {}) or {}
            y_pre = float(spd.get("preprocess", 0.0)) if spd else None
            y_inf = float(spd.get("inference", 0.0)) if spd else None
            y_post = float(spd.get("postprocess", 0.0)) if spd else None

            n_det = 0
            n_det_roi = 0
            emb_total_ms = 0.0
            now = time.time()

            used_gids_this_frame = set()

            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                ids_local = (res.boxes.id.cpu().numpy().astype(int)
                             if hasattr(res.boxes, "id") and res.boxes.id is not None else None)

                # espessura proporcional para bbox
                _, box_thick = _text_style_for(frame, base_factor=0.9)

                for i, box in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    c = clss[i]
                    if class_set is not None and c not in class_set:
                        continue

                    n_det += 1
                    cls_name = names.get(c, str(c))

                    inside = True
                    overlap_ratio = 1.0
                    if args.roi_mode != "none" and roi_mask is not None:
                        overlap_ratio = bbox_roi_overlap_ratio((x1, y1, x2, y2), roi_mask)
                        inside = overlap_ratio >= args.roi_min_intersection

                    color = (50, 220, 50) if inside else (120, 120, 120)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, box_thick))

                    gid_text = f"{cls_name} (ROI {overlap_ratio*100:.0f}%)"
                    if inside:
                        n_det_roi += 1
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                        t_e0 = time.perf_counter()
                        vec = emb.extract(crop)
                        t_e1 = time.perf_counter()
                        emb_total_ms += (t_e1 - t_e0) * 1000.0

                        if vec is not None:
                            with gallery_lock:
                                proposed_gid = None
                                if ids_local is not None:
                                    lid = int(ids_local[i])
                                    if lid in local2global:
                                        proposed_gid = local2global[lid]
                                    else:
                                        proposed_gid, _ = gallery.match_or_create(cls_name, vec, now)
                                    if proposed_gid in used_gids_this_frame:
                                        proposed_gid = gallery.create_new(cls_name, vec, now)
                                    local2global[lid] = proposed_gid
                                    gid = proposed_gid
                                else:
                                    gid, _ = gallery.match_or_create(cls_name, vec, now)
                                    if gid in used_gids_this_frame:
                                        gid = gallery.create_new(cls_name, vec, now)

                                used_gids_this_frame.add(gid)
                                # EMA global (após unicidade)
                                bank = gallery.gallery[cls_name]
                                if gid in bank:
                                    rec = bank[gid]
                                    rec["emb"] = 0.9 * rec["emb"] + 0.1 * vec
                                    rec["emb"] = rec["emb"] / (np.linalg.norm(rec["emb"]) + 1e-12)
                                    rec["last_seen"] = now
                            gid_text = f"{cls_name} GID#{gid}"

                    draw_label(frame, (x1, y1, x2, y2), gid_text)

            # FPS
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_window.append(fps_inst)
            fps_avg = sum(fps_window) / len(fps_window)

            # CUDA mem (global, aproximação)
            cuda_mem_mb = 0.0
            if torch.cuda.is_available() and "cuda" in str(device).lower():
                try:
                    cuda_mem_mb = torch.cuda.memory_allocated() / (1024**2)
                except Exception:
                    pass

            if csv_writer:
                emb_avg_ms = (emb_total_ms / max(n_det_roi, 1))
                csv_writer.writerow([
                    frame_idx, n_det, n_det_roi,
                    f"{y_pre:.3f}" if y_pre is not None else "",
                    f"{y_inf:.3f}" if y_inf is not None else "",
                    f"{y_post:.3f}" if y_post is not None else "",
                    f"{emb_total_ms:.3f}",
                    f"{emb_avg_ms:.3f}",
                    f"{fps_inst:.3f}", f"{fps_avg:.3f}",
                    f"{cuda_mem_mb:.1f}"
                ])

            # HUD proporcional no canto superior esquerdo
            hud_txt = f"[Cam {idx}] FPS {fps_inst:5.1f} (avg {fps_avg:5.1f})  emb:{emb.name}"
            draw_text_proportional(frame, hud_txt, anchor="tl", margin=10, base_factor=0.95)

            # Publica frame anotado no buffer compartilhado (para mosaico)
            with shared_lock:
                shared_frames[idx] = frame.copy()

            if writer:
                writer.write(frame)
            frame_idx += 1

            if args.print_every > 0 and frame_idx % args.print_every == 0:
                print(f"[Cam {idx}][{frame_idx:04d}] det:{n_det} roi:{n_det_roi}  yolo_inf(ms):{y_inf if y_inf is not None else 'NA'}  "
                      f"emb_total(ms):{emb_total_ms:.1f}  fps(avg):{fps_avg:.2f}  cuda_mem(MB):{cuda_mem_mb:.1f}")

    finally:
        if writer: writer.release()
        if csv_file: csv_file.close()
        print(f"[Cam {idx}] Finalizado.")


# =========================
# Principal (multi-source)
# =========================
def parse_sources(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return parts

def maybe_split_arg_by_comma(s: Optional[str]) -> List[Optional[str]]:
    if s is None:
        return []
    parts = [p.strip() for p in s.split(",")]
    return parts

def prepare_rois_for_sources(
    sources: List[str],
    args: argparse.Namespace
) -> List[List[Tuple[int,int]]]:
    """
    Retorna lista de polígonos por câmera (cada item = lista de pontos [(x,y), ...] * N polígonos).
    Para modo 'interactive', abre sequência de editores (um por câmera) ANTES de iniciar as threads.
    Para 'file', aceita --roi-file como lista separada por vírgula (um por câmera) ou único arquivo para todas.
    """
    rois_all = [[] for _ in sources]

    if args.roi_mode == "none":
        return rois_all

    # capturar 1 frame de cada fonte para desenhar ROI (se interactive)
    if args.roi_mode == "interactive":
        for i, src in enumerate(sources):
            cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                print(f"[ROI] (Cam {i}) não conseguiu capturar 1º frame para ROI. Desabilitando ROI nessa câmera.")
                continue
            editor = ROIEditor(frame, window_suffix=f" (Cam {i})")
            polys = editor.loop()
            rois_all[i] = polys
            if args.roi_save:
                roi_save_path = args.roi_save.replace("{idx}", str(i))
                try:
                    Path(roi_save_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(roi_save_path, "w") as f:
                        json.dump(polys, f)
                    print(f"[ROI] (Cam {i}) ROIs salvas em {roi_save_path}")
                except Exception as e:
                    print(f"[ROI] (Cam {i}) Erro ao salvar ROIs: {e}")

    elif args.roi_mode == "file":
        roi_files = maybe_split_arg_by_comma(args.roi_file) if args.roi_file else []
        if len(roi_files) == 1 and len(sources) > 1:
            path = roi_files[0]
            try:
                with open(path, "r") as f:
                    polys = json.load(f)
                polys = [[tuple(map(int, p)) for p in poly] for poly in polys]
                for i in range(len(sources)):
                    rois_all[i] = polys
                print(f"[ROI] Arquivo único aplicado a todas as câmeras: {path}")
            except Exception as e:
                print(f"[ROI] Falha ao ler {path}. Desabilitando ROI.")
        else:
            for i, path in enumerate(roi_files):
                if not path:
                    continue
                try:
                    with open(path, "r") as f:
                        polys = json.load(f)
                    polys = [[tuple(map(int, p)) for p in poly] for poly in polys]
                    rois_all[i] = polys
                    print(f"[ROI] (Cam {i}) Carregado de {path}. Polígonos: {len(polys)}")
                except Exception as e:
                    print(f"[ROI] (Cam {i}) Falha ao ler JSON {path}: {e}. ROI desabilitada para essa câmera.")

    return rois_all


def main():
    ap = argparse.ArgumentParser(description="Multi-Source Persistent ID Tracking with Global ReID + ROIs + Mosaic (textos proporcionais)")
    # ===== multi-sources =====
    ap.add_argument("--sources", type=str, required=True,
                    help='Lista separada por vírgulas, ex.: "0,1,/path/video.mp4,rtsp://..."')
    ap.add_argument("--weights", type=str, default="yolo11n.pt")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml")
    ap.add_argument("--device", type=str, default=None, help="cuda:0 ou cpu (None = auto)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--class-filter", type=str, default=None)
    ap.add_argument("--match-thresh", type=float, default=0.6)
    ap.add_argument("--gallery-ttl", type=float, default=600.0)
    ap.add_argument("--emb-backbone", type=str, default="resnet50",
                    choices=["resnet50","resnet18","mobilenetv3_small","mobilenetv3_large"])

    # padrões com placeholder {idx} (substituído por 0,1,2,...)
    ap.add_argument("--save", type=str, default=None,
                    help="Arquivo MP4 de saída por câmera, use {idx}, ex.: out_cam{idx}.mp4")
    ap.add_argument("--log-csv", type=str, default=None,
                    help="CSV de log por câmera, use {idx}, ex.: logs/run_cam{idx}.csv")
    ap.add_argument("--print-every", type=int, default=30)

    # === argumentos de ROI ===
    ap.add_argument("--roi-mode", type=str, default="interactive", choices=["interactive", "file", "none"],
                    help="interactive = desenhar por câmera antes de iniciar; file = carregar JSON; none = desabilitar")
    ap.add_argument("--roi-file", type=str, default=None,
                    help="JSON(s) com lista de polígonos. Único arquivo para todas OU lista separada por vírgulas para cada câmera.")
    ap.add_argument("--roi-save", type=str, default=None,
                    help="Se definido e interactive, salva as ROIs em JSON por câmera (use {idx})")
    ap.add_argument("--roi-min-intersection", type=float, default=0.3,
                    help="Fração mínima do bbox dentro da ROI para acionar tracking/ReID")

    # === parâmetros do mosaico ===
    ap.add_argument("--tile-w", type=int, default=640, help="Largura de cada tile no mosaico")
    ap.add_argument("--tile-h", type=int, default=360, help="Altura de cada tile no mosaico")
    ap.add_argument("--mosaic-cols", type=int, default=0, help="Colunas do mosaico (0 = auto sqrt)")

    args = ap.parse_args()

    sources = parse_sources(args.sources)
    if not sources:
        print("[ERRO] Nenhuma fonte válida em --sources.")
        return

    # Galeria Global ReID compartilhada + lock
    gallery = GlobalReIDGallery(match_thresh=args.match_thresh, ttl_seconds=args.gallery_ttl, ema=0.9)
    gallery_lock = Lock()

    # Prepara ROIs por câmera antes de iniciar threads
    rois_all = prepare_rois_for_sources(sources, args)

    # Buffers compartilhados para mosaico
    shared_frames = [None] * len(sources)
    shared_lock = Lock()
    stop_event = Event()

    threads: List[Thread] = []
    try:
        for i, src in enumerate(sources):
            t = Thread(
                target=process_stream,
                args=(i, src, args, gallery, gallery_lock, rois_all[i],
                      shared_frames, shared_lock, stop_event),
                daemon=True
            )
            t.start()
            threads.append(t)

        # Loop principal: renderiza mosaico único e captura teclado
        try:
            cv2.namedWindow("MultiCam Mosaic", cv2.WINDOW_NORMAL)
        except Exception:
            cv2.namedWindow("MultiCam Mosaic")

        labels = [f"Cam {i}" for i in range(len(sources))]
        cols = args.mosaic_cols if args.mosaic_cols > 0 else int(math.ceil(math.sqrt(len(sources))))
        tile_w, tile_h = args.tile_w, args.tile_h

        while True:
            with shared_lock:
                frames_copy = [f.copy() if f is not None else None for f in shared_frames]
            mosaic = build_mosaic(frames_copy, cols, tile_w, tile_h, labels)
            cv2.imshow("MultiCam Mosaic", mosaic)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_event.set()
                break

        for t in threads:
            t.join()

    except KeyboardInterrupt:
        print("\n[MAIN] Interrompido pelo usuário (Ctrl+C).")
        stop_event.set()
        for t in threads:
            t.join()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[MAIN] Finalizado.")


if __name__ == "__main__":
    main()
