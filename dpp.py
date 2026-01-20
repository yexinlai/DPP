#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mp3_box_full_monotonic.py
------------------------------------------------------------
Full sequence MP3-like propagation with SAM2 + GroundingDINO boxes (ONE CSV for all frames)

Fixes / Design goals:
  1) Frames may have NO boxes -> never call propagate_in_video() with empty prompts.
  2) Avoid infinite oscillation between neighboring frames with no boxes:
     - Anchor selection is forward-only (monotonic). Never move anchor backward.
     - Guarantee anchor index strictly increases across iterations.
  3) GitHub-ready:
     - No hard-coded absolute paths; everything is CLI-configurable.
     - Clear logging, robust IO, type hints, and structured code.
------------------------------------------------------------
CSV format (GroundingDINO outputs):
  required columns: image_name, xmin, ymin, xmax, ymax, label
  optional: score|conf|confidence|prob

Output:
  - merged masks (single label map per frame) under <out_dir>/merged
  - per-class binary masks under <out_dir>/per_class/clsXX

Usage example:
  python mp3_box_full_monotonic.py \
    --ct_dir data/amos/image \
    --dino_csv data/amos/box/bboxes.csv \
    --sam_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam_ckpt checkpoints/sam2.1_hiera_large.pt \
    --out_dir outputs/run1 \
    --jpeg_cache .cache/jpeg_amos \
    --log_dir logs \
    --step 5 --bin_thr 0.5 --dsc_thr 0.6
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor


# ============================================================
# Config
# ============================================================
@dataclass
class RunConfig:
    ct_dir: str
    dino_csv: str
    out_dir: str
    jpeg_cache: str
    log_dir: str

    sam_cfg: str
    sam_ckpt: str
    device: str = "cuda"

    dino_img_dir: Optional[str] = None
    min_label: int = 1
    dino_min_score: float = -1.0

    step: int = 5
    bin_thr: float = 0.5
    dsc_thr: float = 0.6

    max_lookahead: int = 1000
    autocast_bf16: bool = True


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(
        description="MP3-like full-sequence propagation using SAM2 + GroundingDINO box prompts (one CSV)."
    )
    p.add_argument("--ct_dir", required=True, help="Directory containing CT frames (png/jpg/tif...).")
    p.add_argument("--dino_csv", required=True, help="GroundingDINO CSV containing detections for all frames.")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--jpeg_cache", default=".cache/jpeg_cache", help="JPEG cache directory.")
    p.add_argument("--log_dir", default="logs", help="Logging directory.")

    p.add_argument("--sam_cfg", required=True, help="SAM2 config yaml path.")
    p.add_argument("--sam_ckpt", required=True, help="SAM2 checkpoint path.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for SAM2 predictor.")

    p.add_argument("--dino_img_dir", default=None, help="If DINO ran on different resolution, provide that image dir for rescale.")
    p.add_argument("--min_label", type=int, default=1, help="Minimum class id to keep from CSV.")
    p.add_argument("--dino_min_score", type=float, default=-1.0, help="Minimum score threshold if score column exists.")

    p.add_argument("--step", type=int, default=5, help="Propagation half-window (left/right) around anchor.")
    p.add_argument("--bin_thr", type=float, default=0.5, help="Binarization threshold for prob maps.")
    p.add_argument("--dsc_thr", type=float, default=0.6, help="Anomaly DSC threshold.")
    p.add_argument("--max_lookahead", type=int, default=1000, help="Max forward search distance for next prompt frame.")
    p.add_argument("--autocast_bf16", action="store_true", help="Enable torch.autocast(bfloat16) for CUDA.")
    p.add_argument("--no_autocast_bf16", dest="autocast_bf16", action="store_false", help="Disable bf16 autocast.")
    p.set_defaults(autocast_bf16=True)

    args = p.parse_args()
    return RunConfig(
        ct_dir=args.ct_dir,
        dino_csv=args.dino_csv,
        out_dir=args.out_dir,
        jpeg_cache=args.jpeg_cache,
        log_dir=args.log_dir,
        sam_cfg=args.sam_cfg,
        sam_ckpt=args.sam_ckpt,
        device=args.device,
        dino_img_dir=args.dino_img_dir,
        min_label=args.min_label,
        dino_min_score=args.dino_min_score,
        step=args.step,
        bin_thr=args.bin_thr,
        dsc_thr=args.dsc_thr,
        max_lookahead=args.max_lookahead,
        autocast_bf16=args.autocast_bf16,
    )


# ============================================================
# Logging
# ============================================================
def setup_logger(log_dir: str, name: str = "mp3_box_full_monotonic") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    return logger


# ============================================================
# Utils
# ============================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: str, exts: Tuple[str, ...] = IMG_EXTS) -> List[str]:
    paths: List[str] = []
    for e in exts:
        paths += glob.glob(os.path.join(folder, f"*{e}"))
        paths += glob.glob(os.path.join(folder, f"*{e.upper()}"))
    return sorted(paths)


def ensure_jpeg_folder_from_ct(ct_dir: str, jpeg_dir: str) -> Tuple[str, List[str]]:
    """
    Convert CT frames to numeric JPEG cache (000000.jpg, 000001.jpg, ...)
    Return:
      jpeg_dir, basenames (original CT filenames)
    """
    os.makedirs(jpeg_dir, exist_ok=True)
    ct_paths = list_images(ct_dir)
    if not ct_paths:
        raise RuntimeError(f"No CT frames found in: {ct_dir}")

    basenames = [os.path.basename(p) for p in ct_paths]
    existing = sorted(glob.glob(os.path.join(jpeg_dir, "*.jpg")))

    # Reuse cache only if counts match; else rebuild.
    if len(existing) == len(ct_paths):
        return jpeg_dir, basenames

    for p in existing:
        os.remove(p)

    for i, pth in enumerate(ct_paths):
        img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {pth}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out = os.path.join(jpeg_dir, f"{i:06d}.jpg")
        ok = cv2.imwrite(out, img)
        if not ok:
            raise RuntimeError(f"Failed to write jpeg: {out}")

    return jpeg_dir, basenames


def dice_coef(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int((a & b).sum())
    denom = int(a.sum() + b.sum())
    return float((2 * inter + eps) / (denom + eps))


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ============================================================
# GroundingDINO Provider (ONE CSV for ALL frames)
# ============================================================
class GroundingDinoBoxPromptProviderFromOneCSV:
    """
    One CSV contains detections for all frames:
      required columns:
        image_name, xmin, ymin, xmax, ymax, label
      optional:
        score / conf / confidence / prob

    For each image_name and each class(label), keep one "best" box:
      - if score exists: keep max score
      - else: keep max area
    """

    def __init__(
        self,
        csv_path: str,
        basenames: List[str],
        dino_img_dir: Optional[str] = None,
        min_label: int = 1,
        min_score: float = -1.0,
    ):
        self.csv_path = csv_path
        self.basenames = basenames
        self.dino_img_dir = dino_img_dir
        self.min_label = int(min_label)
        self.min_score = float(min_score)

        # key(image_name in csv) -> cls_id -> (key_val, xyxy)
        self._index: Dict[str, Dict[int, Tuple[float, np.ndarray]]] = {}
        self._stem_to_key: Dict[str, str] = {}
        self._build_index()

    def _build_index(self) -> None:
        if not os.path.exists(self.csv_path):
            raise RuntimeError(f"CSV not found: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError("Empty CSV header")

            fields = {k.strip().lower(): k for k in reader.fieldnames}

            def need(name: str) -> str:
                if name not in fields:
                    raise RuntimeError(f"CSV missing column `{name}`. Columns={reader.fieldnames}")
                return fields[name]

            k_img = need("image_name")
            k_x1 = need("xmin")
            k_y1 = need("ymin")
            k_x2 = need("xmax")
            k_y2 = need("ymax")
            k_cls = need("label")

            k_score = None
            for cand in ["score", "conf", "confidence", "prob"]:
                if cand in fields:
                    k_score = fields[cand]
                    break

            for row in reader:
                img_name_raw = str(row.get(k_img, "")).strip()
                if not img_name_raw:
                    continue

                img_key = img_name_raw
                img_base = os.path.basename(img_name_raw)
                stem = os.path.splitext(img_base)[0]

                try:
                    x1 = float(row[k_x1]); y1 = float(row[k_y1])
                    x2 = float(row[k_x2]); y2 = float(row[k_y2])
                    cls = int(float(row[k_cls]))
                except Exception:
                    continue

                if cls < self.min_label:
                    continue

                xyxy = np.array([x1, y1, x2, y2], dtype=np.float32)
                if xyxy[2] < xyxy[0]:
                    xyxy[0], xyxy[2] = xyxy[2], xyxy[0]
                if xyxy[3] < xyxy[1]:
                    xyxy[1], xyxy[3] = xyxy[3], xyxy[1]

                if (xyxy[2] - xyxy[0]) < 1 or (xyxy[3] - xyxy[1]) < 1:
                    continue

                if k_score is not None:
                    try:
                        sc = float(row[k_score])
                    except Exception:
                        sc = 0.0
                    if sc < self.min_score:
                        continue
                    key_val = sc
                else:
                    key_val = float((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))  # area

                if img_key not in self._index:
                    self._index[img_key] = {}

                # Record a mapping from stem to a representative csv key.
                self._stem_to_key.setdefault(stem, img_key)

                if (cls not in self._index[img_key]) or (key_val > self._index[img_key][cls][0]):
                    self._index[img_key][cls] = (key_val, xyxy)

        if not self._index:
            raise RuntimeError(f"No valid detections parsed from CSV: {self.csv_path}")

    def _read_image_hw(self, img_dir: str, basename_or_key: str) -> Tuple[int, int]:
        base = os.path.basename(basename_or_key)
        p = os.path.join(img_dir, base)

        if not os.path.exists(p):
            stem = os.path.splitext(base)[0]
            for ext in IMG_EXTS:
                pp = os.path.join(img_dir, stem + ext)
                if os.path.exists(pp):
                    p = pp
                    break

        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            raise RuntimeError(f"Failed to read image for size: {p}")
        return int(im.shape[0]), int(im.shape[1])

    def _lookup_key_for_basename(self, basename: str) -> Optional[str]:
        stem = os.path.splitext(basename)[0]

        # Try exact basename matches first.
        for k in self._index.keys():
            if os.path.basename(k) == basename:
                return k

        # Then try stem matches.
        for k in self._index.keys():
            if os.path.splitext(os.path.basename(k))[0] == stem:
                return k

        # Fallback stem->key map
        return self._stem_to_key.get(stem, None)

    def classes_in_frame(self, t: int) -> List[int]:
        basename = self.basenames[t]
        key = self._lookup_key_for_basename(basename)
        if key is None:
            return []
        return sorted(list(self._index[key].keys()))

    def get_box_xyxy(
        self,
        t: int,
        cls: int,
        target_hw: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        basename = self.basenames[t]
        key = self._lookup_key_for_basename(basename)
        if key is None:
            return None

        cls = int(cls)
        if cls not in self._index[key]:
            return None

        box = self._index[key][cls][1].copy().astype(np.float32)

        # Rescale box if DINO ran on different resolution.
        if target_hw is not None and self.dino_img_dir is not None:
            dh, dw = self._read_image_hw(self.dino_img_dir, key)
            th, tw = target_hw
            if (dh, dw) != (th, tw):
                sx = tw / float(dw)
                sy = th / float(dh)
                box[0] *= sx; box[2] *= sx
                box[1] *= sy; box[3] *= sy

        # Clamp to target size.
        if target_hw is not None:
            th, tw = target_hw
            box[0] = np.clip(box[0], 0, tw - 1)
            box[2] = np.clip(box[2], 0, tw - 1)
            box[1] = np.clip(box[1], 0, th - 1)
            box[3] = np.clip(box[3], 0, th - 1)

        if (box[2] - box[0]) < 1 or (box[3] - box[1]) < 1:
            return None

        return box


# ============================================================
# Cache / anomaly
# ============================================================
BestCache = Dict[int, Dict[int, np.ndarray]]  # t -> cls -> prob_map


def merge_into_best_cache(best: BestCache, new: BestCache) -> None:
    for t, fmap in new.items():
        if t not in best:
            best[t] = {c: p.copy() for c, p in fmap.items()}
        else:
            for c, p in fmap.items():
                if c in best[t]:
                    best[t][c] = np.maximum(best[t][c], p)
                else:
                    best[t][c] = p.copy()


def is_anomaly(cache_map: Dict[int, np.ndarray], cur_map: Dict[int, np.ndarray], thr: float, dsc_thr: float) -> bool:
    C = set(cache_map.keys())
    P = set(cur_map.keys())
    if C ^ P:
        return True
    for c in C & P:
        if dice_coef(cache_map[c] > thr, cur_map[c] > thr) < dsc_thr:
            return True
    return False


# ============================================================
# Prompt availability + monotonic anchor search
# ============================================================
def frame_has_prompts(provider: GroundingDinoBoxPromptProviderFromOneCSV, t: int) -> bool:
    try:
        return len(provider.classes_in_frame(t)) > 0
    except Exception:
        return False


def find_next_anchor_with_prompts(
    provider: GroundingDinoBoxPromptProviderFromOneCSV,
    a: int,
    T: int,
    max_lookahead: int = 1000
) -> Optional[int]:
    """
    Forward-only: find first t >= a with prompts, within lookahead. Never returns < a.
    """
    if a < 0:
        a = 0
    end = min(T - 1, a + max_lookahead)
    for t in range(a, end + 1):
        if frame_has_prompts(provider, t):
            return t
    return None


# ============================================================
# SAM2 helpers
# ============================================================
@torch.inference_mode()
def init_state(
    predictor,
    jpeg_dir: str,
    provider: GroundingDinoBoxPromptProviderFromOneCSV,
    t: int
) -> Optional[object]:
    """
    Initialize SAM2 video state and add box prompts at frame t.
    IMPORTANT: If no valid box is added, return None (never propagate with empty prompts).
    """
    st = predictor.init_state(video_path=jpeg_dir)

    jpg_path = os.path.join(jpeg_dir, f"{t:06d}.jpg")
    im = cv2.imread(jpg_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read jpeg frame: {jpg_path}")
    th, tw = int(im.shape[0]), int(im.shape[1])

    cls_list = provider.classes_in_frame(t)

    added = 0
    for c in cls_list:
        box = provider.get_box_xyxy(t, c, target_hw=(th, tw))
        if box is None:
            continue

        if hasattr(predictor, "add_new_points_or_box"):
            predictor.add_new_points_or_box(
                st,
                frame_idx=t,
                obj_id=int(c),
                points=None,
                labels=None,
                box=box
            )
        elif hasattr(predictor, "add_new_box"):
            predictor.add_new_box(st, t, int(c), box)
        else:
            raise RuntimeError("Predictor does not support box prompting (no add_new_points_or_box/add_new_box).")

        added += 1

    if added == 0:
        return None

    return st


@torch.inference_mode()
def run_forward(
    predictor,
    state,
    a: int,
    R: int,
    best_snapshot: BestCache,
    detect: bool,
    thr: float,
    dsc_thr: float,
) -> Tuple[BestCache, Optional[int]]:
    out: BestCache = {}
    anomaly: Optional[int] = None
    max_track = R - a + 1

    for t, oids, masks in predictor.propagate_in_video(
        state, start_frame_idx=a, max_frame_num_to_track=max_track, reverse=False
    ):
        t = int(t)
        fmap: Dict[int, np.ndarray] = {}
        for k, c in enumerate(oids):
            prob = torch.sigmoid(masks[k]).detach().cpu().numpy().squeeze()
            fmap[int(c)] = prob
        out[t] = fmap
        del masks

        if detect and t in best_snapshot and t != a:
            if is_anomaly(best_snapshot[t], fmap, thr, dsc_thr):
                anomaly = t
                break

    return out, anomaly


@torch.inference_mode()
def run_reverse_lock(predictor, state, a: int, L: int) -> BestCache:
    out: BestCache = {}
    max_track = a - L + 1

    for t, oids, masks in predictor.propagate_in_video(
        state, start_frame_idx=a, max_frame_num_to_track=max_track, reverse=True
    ):
        t = int(t)
        fmap: Dict[int, np.ndarray] = {}
        for k, c in enumerate(oids):
            prob = torch.sigmoid(masks[k]).detach().cpu().numpy().squeeze()
            fmap[int(c)] = prob
        out[t] = fmap
        del masks

    return out


@torch.inference_mode()
def run_reverse_detect(
    predictor,
    state,
    a: int,
    L: int,
    best_snapshot: BestCache,
    thr: float,
    dsc_thr: float,
) -> Tuple[BestCache, Optional[int]]:
    out: BestCache = {}
    anomaly: Optional[int] = None
    max_track = a - L + 1

    for t, oids, masks in predictor.propagate_in_video(
        state, start_frame_idx=a, max_frame_num_to_track=max_track, reverse=True
    ):
        t = int(t)
        fmap: Dict[int, np.ndarray] = {}
        for k, c in enumerate(oids):
            prob = torch.sigmoid(masks[k]).detach().cpu().numpy().squeeze()
            fmap[int(c)] = prob
        out[t] = fmap
        del masks

        if t in best_snapshot:
            if is_anomaly(best_snapshot[t], fmap, thr, dsc_thr):
                anomaly = t
                break

    return out, anomaly


# ============================================================
# Save Results
# ============================================================
def save_merged(out_dir: str, basenames: List[str], best: BestCache, thr: float) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for t, fmap in best.items():
        cls = sorted(fmap.keys())
        if not cls:
            continue

        stack = np.stack([fmap[c] for c in cls])  # [C,H,W]
        pmax = stack.max(0)
        idx = stack.argmax(0)

        lab = np.zeros(pmax.shape, np.int32)
        fg = pmax > thr
        lab[fg] = np.array(cls)[idx[fg]]

        out_path = os.path.join(out_dir, basenames[t])
        cv2.imwrite(out_path, lab.astype(np.uint8))


def save_per_class(out_dir: str, basenames: List[str], best: BestCache, thr: float) -> None:
    all_cls = set()
    for fmap in best.values():
        all_cls |= set(fmap.keys())
    for c in all_cls:
        os.makedirs(os.path.join(out_dir, f"cls{c:02d}"), exist_ok=True)

    for t, fmap in best.items():
        for c, p in fmap.items():
            m = (p > thr).astype(np.uint8) * 255
            out_path = os.path.join(out_dir, f"cls{c:02d}", basenames[t])
            cv2.imwrite(out_path, m)


# ============================================================
# Main pipeline
# ============================================================
def build_predictor(cfg: RunConfig):
    predictor = build_sam2_video_predictor(cfg.sam_cfg, cfg.sam_ckpt)
    if cfg.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA selected but not available. Use --device cpu or install CUDA correctly.")
        predictor = predictor.to("cuda")
    else:
        predictor = predictor.to("cpu")
    predictor.eval()
    return predictor


def main():
    cfg = parse_args()
    logger = setup_logger(cfg.log_dir)

    logger.info("========== Run Config ==========")
    for k, v in asdict(cfg).items():
        logger.info(f"{k}: {v}")
    logger.info("================================")

    if cfg.device == "cuda" and cfg.autocast_bf16:
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    jpeg_dir, basenames = ensure_jpeg_folder_from_ct(cfg.ct_dir, cfg.jpeg_cache)
    T = len(basenames)
    logger.info(f"[Info] Total frames: {T}")

    provider = GroundingDinoBoxPromptProviderFromOneCSV(
        csv_path=cfg.dino_csv,
        basenames=basenames,
        dino_img_dir=cfg.dino_img_dir,
        min_label=cfg.min_label,
        min_score=cfg.dino_min_score,
    )

    chkN = min(T, 50)
    cnt = sum(1 for i in range(chkN) if frame_has_prompts(provider, i))
    logger.info(f"[Check] Frames with boxes in first {chkN}: {cnt}")

    predictor = build_predictor(cfg)

    best_cache: BestCache = {}

    # --------------------------------------------------------
    # Stage 0: pick first prompt frame globally (forward-only)
    # --------------------------------------------------------
    a0 = find_next_anchor_with_prompts(provider, 0, T, max_lookahead=T)
    if a0 is None:
        raise RuntimeError("No frames have any detected boxes. Cannot run SAM2 propagation.")
    if a0 != 0:
        logger.info(f"[Anchor] First prompt frame is {a0} (frame 0 has no boxes).")

    st0 = init_state(predictor, jpeg_dir, provider, a0)
    if st0 is None:
        raise RuntimeError(f"Unexpected: anchor frame {a0} has no valid boxes after filtering.")

    fwd0, _ = run_forward(
        predictor, st0,
        a0, min(T - 1, a0 + cfg.step),
        best_snapshot={}, detect=False,
        thr=cfg.bin_thr, dsc_thr=cfg.dsc_thr
    )
    merge_into_best_cache(best_cache, fwd0)
    del st0, fwd0
    cuda_cleanup()

    # --------------------------------------------------------
    # Stage 1: start from a0+STEP, align to next prompt frame (forward-only)
    # --------------------------------------------------------
    step = int(cfg.step)
    a = min(T - 1, a0 + step)
    a = find_next_anchor_with_prompts(provider, a, T, max_lookahead=max(step * 5, 50))
    if a is None:
        logger.info("[Stop] No more prompt frames to the right after stage0.")
        save_merged(os.path.join(cfg.out_dir, "merged"), basenames, best_cache, cfg.bin_thr)
        save_per_class(os.path.join(cfg.out_dir, "per_class"), basenames, best_cache, cfg.bin_thr)
        logger.info("DONE")
        return

    st1 = init_state(predictor, jpeg_dir, provider, a)
    if st1 is not None:
        snap = {t: {c: p.copy() for c, p in fmap.items()} for t, fmap in best_cache.items()}
        _, anomaly = run_reverse_detect(
            predictor, st1, a, L=a0,
            best_snapshot=snap, thr=cfg.bin_thr, dsc_thr=cfg.dsc_thr
        )
        del st1, snap
        cuda_cleanup()
    else:
        anomaly = None
        cuda_cleanup()

    if anomaly is not None:
        logger.info(f"[Anomaly] Found anomaly at frame {anomaly}, re-anchoring (forward-only).")
        a = find_next_anchor_with_prompts(provider, anomaly, T, max_lookahead=max(step * 5, 50))
        if a is None:
            logger.info("[Stop] No prompt frames after anomaly.")
            save_merged(os.path.join(cfg.out_dir, "merged"), basenames, best_cache, cfg.bin_thr)
            save_per_class(os.path.join(cfg.out_dir, "per_class"), basenames, best_cache, cfg.bin_thr)
            logger.info("DONE")
            return

        step = max(1, min(step, a))
        stA = init_state(predictor, jpeg_dir, provider, a)
        if stA is not None:
            rev = run_reverse_lock(predictor, stA, a, max(0, a - step))
            merge_into_best_cache(best_cache, rev)
            del stA, rev
            cuda_cleanup()
        else:
            cuda_cleanup()

    # --------------------------------------------------------
    # Main loop: monotonic forward progression
    # --------------------------------------------------------
    last_a = a0  # last committed anchor (monotonic)

    while True:
        # Enforce strict forward progress
        if a <= last_a:
            a = last_a + 1
        if a >= T:
            break

        # Align to next prompt frame forward-only
        a2 = find_next_anchor_with_prompts(provider, a, T, max_lookahead=max(step * 5, 50))
        if a2 is None:
            logger.info("[Stop] No further prompt frames; finishing.")
            break

        if a2 != a:
            logger.info(f"[Anchor] Advance anchor {a} -> {a2} due to missing boxes.")
        a = a2
        last_a = a  # commit

        L = max(0, a - step)
        R = min(T - 1, a + step)

        # reverse lock
        st_r = init_state(predictor, jpeg_dir, provider, a)
        if st_r is not None:
            rev = run_reverse_lock(predictor, st_r, a, L)
            merge_into_best_cache(best_cache, rev)
            del st_r, rev
            cuda_cleanup()
        else:
            logger.warning(f"[Warn] init_state returned None at anchor {a} (reverse skipped).")
            cuda_cleanup()

        # forward detect
        snap = {t: {c: p.copy() for c, p in fmap.items()} for t, fmap in best_cache.items()}
        st_f = init_state(predictor, jpeg_dir, provider, a)
        if st_f is None:
            logger.warning(f"[Warn] init_state returned None at anchor {a} (forward skipped).")
            del snap
            cuda_cleanup()
            a = a + 1
            continue

        fwd, anomaly = run_forward(
            predictor, st_f, a, R, snap,
            detect=True, thr=cfg.bin_thr, dsc_thr=cfg.dsc_thr
        )
        merge_into_best_cache(best_cache, fwd)
        del st_f, fwd, snap
        cuda_cleanup()

        if anomaly is None:
            if R >= T - 1:
                break
            a = a + step
        else:
            logger.info(f"[Anomaly] Found at frame {anomaly}; re-anchor forward-only.")
            a = anomaly
            # conservative step adjustment (keeps your original style, but avoids pathological behavior)
            step = max(1, min(step, last_a))

    # Save outputs
    save_merged(os.path.join(cfg.out_dir, "merged"), basenames, best_cache, cfg.bin_thr)
    save_per_class(os.path.join(cfg.out_dir, "per_class"), basenames, best_cache, cfg.bin_thr)
    logger.info("DONE")


if __name__ == "__main__":
    main()
