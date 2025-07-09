#!/usr/bin/env python3
"""
viz_predictions_viewer.py

Wyświetla obrazy z folderu i nakłada na nie ramki z predictions.json.
* Brak zapisu na dysk.
* Stałe (skalowalne) okno o zadanych wymiarach.
* Spacja = następny obraz; Esc = wyjście.

Wymagane: pip install opencv-python numpy
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import cv2
import numpy as np

# ---------- argumenty ----------
parser = argparse.ArgumentParser(description="Podgląd predykcji na obrazach (tylko wyświetlanie)")
parser.add_argument("--predictions_file", default="data/predictions.json",
                    help="Ścieżka do pliku predictions.json")
parser.add_argument("--images_dir", required=False, default="data/images",
                    help="Folder z obrazami (JPG/PNG/TIFF...)")
parser.add_argument("--win", nargs=2, type=int, metavar=("SZER", "WYS"),
                    default=(1280, 720), help="Rozmiar okna w pikselach, domyślnie 1280 × 720")
args = parser.parse_args()

WIN_W, WIN_H = args.win
cv2.namedWindow("predictions", cv2.WINDOW_NORMAL)
cv2.resizeWindow("predictions", WIN_W, WIN_H)

# ---------- wczytaj predykcje ----------
pred_path = Path(args.predictions_file)
with pred_path.open("r", encoding="utf-8") as f:
    preds = json.load(f)

by_image: dict[str, list[dict]] = {}
for p in preds:
    by_image.setdefault(p["image_id"], []).append(p)

# ---------- pomocnicze ----------
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def color_for(cat_id: int) -> tuple[int, int, int]:
    """Deterministyczny kolor na podstawie category_id."""
    rng = np.random.RandomState(cat_id)
    return tuple(int(x) for x in rng.randint(0, 255, size=3))

def letterbox(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Skaluje obraz z zachowaniem proporcji i dokłada pasy, aby trafić w docelowy rozmiar okna."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas

# ---------- główna pętla ----------
images = [p for p in sorted(Path(args.images_dir).iterdir())
          if p.suffix.lower() in SUPPORTED]

idx = 0
while idx < len(images):
    img_path = images[idx]
    image_id = img_path.stem
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] Nie można odczytać {img_path}, pomijam.")
        idx += 1
        continue

    # narysuj predykcje
    for pred in by_image.get(image_id, []):
        x, y, w, h = map(int, pred["bbox"])
        cat_id     = pred["category_id"]
        score      = pred["score"]
        colour     = color_for(cat_id)

        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
        label = f"{cat_id}:{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), colour, -1)
        cv2.putText(img, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # dopasuj do okna i pokaż
    shown = letterbox(img, WIN_W, WIN_H)
    cv2.imshow("predictions", shown)

    key = cv2.waitKey(0) & 0xFF
    if key == 27:       # Esc
        break
    elif key == 32:     # spacja
        idx += 1
    # dowolny inny klawisz: pozostaje na tym samym obrazie

cv2.destroyAllWindows()
