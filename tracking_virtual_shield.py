#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Persistent ID Tracking + Global ReID (com profiling) + ROIs múltiplas
- YOLO (Ultralytics) + tracker (ByteTrack/BOT/OC/StrongSORT) para IDs locais.
- ReID global via embeddings (torchvision).
- Profiling por frame (CSV): tempos YOLO, tempo de embedding, FPS, memória CUDA.
- Regra de unicidade por frame -> dois bboxes no mesmo frame nunca compartilham o mesmo GID.
- NOVO: Ativa tracking/ReID APENAS quando o bbox do alvo INTERSECTA as ROIs (múltiplos polígonos).
- ROIs: desenho interativo no primeiro frame OU carregadas de JSON.
"""

import argparse
import csv
import json
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO


# ----------------------------
# UI helpers
# ----------------------------
def draw_label(img, box, text):
    x1, y1, x2, y2 = box
    tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cx = int((x1 + x2) / 2); y = max(0, int(y1) - 8)
    cv2.putText(img, text, (cx - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (cx - tw // 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def draw_polygons(img, polygons, color=(20, 180, 255)):
    overlay = img.copy()
    for poly in polygons:
        if len(poly) >= 3:
            cv2.polylines(overlay, [np.array(poly, dtype=np.int32)], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32)], color=(20, 180, 255, 40))
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


# ----------------------------
# Embedding backbones (torchvision)
# ----------------------------
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


# ----------------------------
# Global ReID Gallery
# ----------------------------
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


# ----------------------------
# ROI helpers
# ----------------------------
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
    def __init__(self, frame):
        self.orig = frame.copy()
        self.img = frame.copy()
        self.polygons = []
        self.current = []
        self.done = False
        self.window = "Desenhe ROIs (ENTER fecha, n novo, u desfaz, c limpar, q sair)"

        # Criação de janela + checagem de GUI com tratamento de erro legível
        try:
            try:
                cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
            except Exception:
                cv2.namedWindow(self.window)

            # Teste rápido de GUI
            _test = "_cv2_gui_test_"
            try:
                cv2.namedWindow(_test)
                cv2.destroyWindow(_test)
            except Exception:
                pass

            cv2.setMouseCallback(self.window, self.on_mouse)

        except cv2.error as e:
            msg = (
                "[ROI] Não foi possível criar janela de GUI do OpenCV.\n"
                "Causas comuns: sessão headless/SSH sem X, DISPLAY ausente, Wayland sem plugin.\n"
                "Soluções:\n"
                "  1) Rode local com GUI habilitada (verifique $DISPLAY).\n"
                "  2) Ou use --roi-mode file com um JSON de polígonos.\n"
                "Exemplo JSON: [[ [x1,y1], [x2,y2], ... ], [ ... ]]\n"
                f"Erro original: {e}"
            )
            raise RuntimeError(msg)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))
            self.redraw()

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
        print("[ROI] Controles: clique adiciona ponto | ENTER fecha polígono | n novo | u desfaz | c limpar | q sair")
        while True:
            cv2.imshow(self.window, self.img)
            key = cv2.waitKey(20) & 0xFF
            if key == 13 or key == 10:  # ENTER
                if len(self.current) >= 3:
                    self.polygons.append(self.current.copy())
                    self.current.clear()
                    self.redraw()
                    print(f"[ROI] Polígono adicionado. Total: {len(self.polygons)}.")
                else:
                    print("[ROI] Polígono precisa de pelo menos 3 pontos.")
            elif key == ord('n'):
                if len(self.current) >= 3:
                    self.polygons.append(self.current.copy())
                self.current.clear()
                self.redraw()
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
            self.polygons.append(self.current.copy())
            self.current.clear()
        return self.polygons


# ----------------------------
# Principal
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Persistent ID Tracking with Global ReID + ROIs")
    ap.add_argument("--source", type=str, required=True, help="Índice de câmera (0) ou caminho de vídeo/imagem")
    ap.add_argument("--weights", type=str, default="yolo11n.pt")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml")
    ap.add_argument("--device", type=str, default=None, help="cuda:0 ou cpu (None = auto)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--class-filter", type=str, default=None)
    ap.add_argument("--match-thresh", type=float, default=0.6)
    ap.add_argument("--gallery-ttl", type=float, default=600.0)
    ap.add_argument("--emb-backbone", type=str, default="resnet50",
                    choices=["resnet50","resnet18","mobilenetv3_small","mobilenetv3_large"])
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--log-csv", type=str, default=None)
    ap.add_argument("--print-every", type=int, default=30)

    # === argumentos de ROI ===
    ap.add_argument("--roi-mode", type=str, default="interactive", choices=["interactive", "file", "none"],
                    help="interactive = desenhar no 1º frame; file = carregar JSON; none = desabilitar filtro ROI")
    ap.add_argument("--roi-file", type=str, default=None, help="JSON com lista de polígonos [[(x,y),...], ...]")
    ap.add_argument("--roi-save", type=str, default=None, help="Se definido, salva as ROIs desenhadas em JSON")
    ap.add_argument("--roi-min-intersection", type=float, default=0.3,
                    help="Fração mínima do bbox dentro da ROI para acionar tracking/ReID")

    args = ap.parse_args()

    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    names = model.names

    class_set = None
    if args.class_filter:
        wanted = {c.strip() for c in args.class_filter.split(",")}
        class_set = {i for i, n in names.items() if n in wanted}

    emb = EmbeddingExtractor(args.emb_backbone, device)
    gallery = GlobalReIDGallery(match_thresh=args.match_thresh, ttl_seconds=args.gallery_ttl, ema=0.9)
    local2global = {}

    writer = None
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    csv_writer = None
    csv_file = None
    if args.log_csv:
        Path(args.log_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.log_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["frame_idx","n_det","n_det_roi","yolo_pre_ms","yolo_inf_ms","yolo_post_ms",
                             "emb_total_ms","emb_avg_ms","fps_inst","fps_avg","cuda_mem_mb"])

    # Stream do YOLO (gera frames + boxes + ids locais)
    results = model.track(
        source=args.source, imgsz=args.imgsz, tracker=args.tracker,
        conf=0.25, iou=0.45, device=device, stream=True, persist=True, verbose=False
    )

    # Estado de ROIs
    roi_polygons = []
    roi_mask = None
    roi_ready = (args.roi_mode == "none")

    win = f"Persistent ID Tracking [{emb.name}]"
    print(f"[INFO] Embedding backbone: {emb.name}")
    t_prev = time.perf_counter()
    fps_window = deque(maxlen=60)
    frame_idx = 0

    try:
        for res in results:
            frame = getattr(res, "orig_img", None)
            if frame is None:
                continue
            frame = frame.copy()
            h, w = frame.shape[:2]

            if writer is None and args.save:
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

            # Inicialização das ROIs no 1º frame útil
            if not roi_ready:
                if args.roi_mode == "file":
                    if not args.roi_file or not Path(args.roi_file).exists():
                        print("[ROI] --roi-file não fornecido ou não encontrado. Caindo para modo 'interactive'.")
                        args.roi_mode = "interactive"
                    else:
                        try:
                            with open(args.roi_file, "r") as f:
                                roi_polygons = json.load(f)
                            roi_polygons = [[tuple(map(int, p)) for p in poly] for poly in roi_polygons]
                            roi_mask = build_roi_mask(roi_polygons, (h, w))
                            roi_ready = True
                            print(f"[ROI] Carregado de {args.roi_file}. Polígonos: {len(roi_polygons)}")
                        except Exception as e:
                            print(f"[ROI] Falha ao ler JSON: {e}. Usando modo 'interactive'.")
                            args.roi_mode = "interactive"

                if args.roi_mode == "interactive":
                    try:
                        editor = ROIEditor(frame)
                    except RuntimeError as e:
                        print(str(e))
                        if args.roi_file and Path(args.roi_file).exists():
                            print("[ROI] Caindo para --roi-mode file usando --roi-file fornecido.")
                            with open(args.roi_file, "r") as f:
                                roi_polygons = json.load(f)
                            roi_polygons = [[tuple(map(int, p)) for p in poly] for poly in roi_polygons]
                            roi_mask = build_roi_mask(roi_polygons, (h, w))
                            roi_ready = True
                            print(f"[ROI] Carregado de {args.roi_file}. Polígonos: {len(roi_polygons)}")
                        else:
                            print("[ROI] Interativo indisponível e nenhum --roi-file foi passado. Finalizando.")
                            return
                    else:
                        roi_polygons = editor.loop()
                        if args.roi_save:
                            try:
                                with open(args.roi_save, "w") as f:
                                    json.dump(roi_polygons, f)
                                print(f"[ROI] ROIs salvas em {args.roi_save}")
                            except Exception as e:
                                print(f"[ROI] Erro ao salvar ROIs: {e}")
                        roi_mask = build_roi_mask(roi_polygons, (h, w))
                        roi_ready = True
                        print(f"[ROI] Polígonos definidos: {len(roi_polygons)}")

                if args.roi_mode == "none":
                    roi_ready = True

            # Desenha ROIs
            if args.roi_mode != "none" and roi_polygons:
                draw_polygons(frame, roi_polygons, color=(20, 180, 255))

            spd = getattr(res, "speed", {}) or {}
            y_pre = float(spd.get("preprocess", 0.0)) if spd else None
            y_inf = float(spd.get("inference", 0.0)) if spd else None
            y_post = float(spd.get("postprocess", 0.0)) if spd else None

            n_det = 0
            n_det_roi = 0
            emb_total_ms = 0.0
            now = time.time()

            used_gids_this_frame = set()  # unicidade de GID por frame

            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                ids_local = (res.boxes.id.cpu().numpy().astype(int)
                             if hasattr(res.boxes, "id") and res.boxes.id is not None else None)

                for i, box in enumerate(boxes_xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    c = clss[i]
                    if class_set is not None and c not in class_set:
                        continue

                    n_det += 1
                    cls_name = names.get(c, str(c))

                    # Checagem ROI
                    inside = True
                    overlap_ratio = 1.0
                    if args.roi_mode != "none" and roi_mask is not None:
                        overlap_ratio = bbox_roi_overlap_ratio((x1, y1, x2, y2), roi_mask)
                        inside = overlap_ratio >= args.roi_min_intersection

                    # Cor diferente se fora da ROI (sem Tracking/ReID)
                    color = (50, 220, 50) if inside else (120, 120, 120)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    gid_text = f"{cls_name} (ROI {overlap_ratio*100:.0f}%)"
                    if inside:
                        n_det_roi += 1
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                        t_e0 = time.perf_counter()
                        vec = emb.extract(crop)
                        t_e1 = time.perf_counter()
                        emb_total_ms += (t_e1 - t_e0) * 1000.0

                        if vec is not None:
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
                            gid_text = f"{cls_name} GID#{gid}"

                            # EMA após unicidade
                            bank = gallery.gallery[cls_name]
                            if gid in bank:
                                rec = bank[gid]
                                rec["emb"] = 0.9 * rec["emb"] + 0.1 * vec
                                rec["emb"] = rec["emb"] / (np.linalg.norm(rec["emb"]) + 1e-12)
                                rec["last_seen"] = now

                    draw_label(frame, (x1, y1, x2, y2), gid_text)

            # FPS
            t_now = time.perf_counter()
            dt = t_now - t_prev
            t_prev = t_now
            fps_inst = (1.0 / dt) if dt > 0 else 0.0
            fps_window.append(fps_inst)
            fps_avg = sum(fps_window) / len(fps_window)

            # CUDA mem
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

            # HUD
            cv2.putText(frame, f"FPS {fps_inst:5.1f} (avg {fps_avg:5.1f})  emb:{emb.name}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 3, cv2.LINE_AA)
            cv2.putText(frame, f"FPS {fps_inst:5.1f} (avg {fps_avg:5.1f})  emb:{emb.name}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2, cv2.LINE_AA)

            cv2.imshow(win, frame)
            if writer: writer.write(frame)
            frame_idx += 1

            if args.print_every > 0 and frame_idx % args.print_every == 0:
                print(f"[{frame_idx:04d}] det:{n_det} roi:{n_det_roi}  yolo_inf(ms):{y_inf if y_inf is not None else 'NA'}  "
                      f"emb_total(ms):{emb_total_ms:.1f}  fps(avg):{fps_avg:.2f}  cuda_mem(MB):{cuda_mem_mb:.1f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if writer: writer.release()
        if csv_file: csv_file.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[DONE] Finalizado.")


if __name__ == "__main__":
    main()

