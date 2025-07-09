from __future__ import annotations
import argparse, csv, math, os, sys
from pathlib import Path
from collections import defaultdict
from typing import Tuple

import cv2
import numpy as np
import csv

# Consts
DEF_REF_DIST = 1.3
DEF_MAX_W, DEF_MAX_H = 1600, 900
IMG_W, IMG_H = 5952, 2976
WIN = "Depth viewer"

CLASSES = {i: n for i, n in enumerate([
    "Bathtub", "Chair", "Table", "TV", "Washing Machine", "Cabinet", "Gaming Console", "Sofa", "Speaker", "Fireplace",
    "Bed", "Wardrobe", "Pillow", "Nightstand", "Toilet", "Shower", "Laundry Rack", "Hair Dryer", "Fridge", "Microwave",
    "Dishwasher", "Stove", "Kettle", "Coffe Machine", "Toaster", "Oven", "Lamp", "Air Conditioning", "Computer", "Plant",
    "Window", "Desk", "Door", "Mirror", "Socket", "Sink", "Aquarium", "Painting", "Air Purifier", "Switch", "Boiler",
    "Rug", "Board", "Vase", "Faucet", "Curtain", "Roller Blind", "Shelf", "Fire Extinguisher", "Fan", "Heater",
    "Car", "Phone", "Clock", "Alarm Sensor"], 0)}

# Globals
cursorPos: tuple[int, int] | None = None

def mouseCallback(event: int, x: int, y: int, flags: int, param):
    global cursorPos
    if event == cv2.EVENT_MOUSEMOVE:
        cursorPos = (x, y)

def equiXYToDirectionVector(u: float, v: float,
                            W: int, H: int) -> np.ndarray:
    """
    (u, v)  – współrzędne piksela panoramy equirectangular
    (W, H)  – rozmiar obrazu
    Zwraca wektor jednostkowy (X, Y, Z) zgodnie z osiami:
        • +Y: przód (φ = 0°  na lewym brzegu)
        • +X: prawo (φ = 90° na ¼ szerokości)
        • +Z: góra  (θ = +90° na górze obrazu)
    """
    phi   = (u / W) * 2.0 * math.pi          # 0 … 2π
    theta = 0.5 * math.pi - (v / H) * math.pi   # +π/2 … −π/2

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    sin_p = math.sin(phi)
    cos_p = math.cos(phi)

    #  X = cosθ · sinφ
    #  Y = cosθ · cosφ     ←  +Y dla φ = 0
    #  Z = sinθ
    return np.array([cos_t * sin_p,
                     cos_t * cos_p,
                     sin_t], dtype=np.float32)

def equiXYToPhiThetaInDeg(x: float, y: float,
                          imageWidth: int, imageHeight: int) -> Tuple[float, float]:
    """
    Zwraca (φ°, θ°) w nowych zakresach:
        φ  0°  … 360°   (0° lewy brzeg, 180° środek, 360° prawy)
        θ +90° … −90°   (góra … dół)
    """
    phi_deg   = (x / imageWidth) * 360.0            # 0 … 360
    theta_deg = 90.0 - (y / imageHeight) * 180.0    # +90 … −90
    return phi_deg, theta_deg

def equiXYDepthToXYZ(u: float, v: float, depth_val: float,
                     W: int, H: int) -> Tuple[float, float, float]:
    """
    Zamienia (u,v) + głębokość na współrzędne 3-D w układzie kamery.
    """
    dir_vec = equiXYToDirectionVector(u, v, W, H)   # jednostkowy
    return tuple(float(depth_val * c) for c in dir_vec)  # przeskalowany

# ───────────────────── Depth ─────────────────────

def load_metric(p: Path):
    return np.load(p).astype(np.float32)

def load_relative(png: Path, txt: Path, ref: float):
    d = cv2.imread(str(png), cv2.IMREAD_UNCHANGED).astype(np.float32)
    if d.ndim == 3:
        d = d.mean(2)
    if d.max() > 1.01:
        d /= 255.0
    x1, y1, x2, y2 = np.loadtxt(txt, delimiter=",", dtype=int)
    return d * (ref / d[y1:y2, x1:x2].mean())

# ───────────────────── BBox ─────────────────────

def bbox_px(b, W, H):
    x, y, w, h = b
    if max(b) <= 1.01:
        x *= W; y *= H; w *= W; h *= H
    x -= w / 2; y -= h / 2
    return int(x), int(y), int(x + w), int(y + h)

# ─────────────────── Predictions ────────────────

def load_yolo(dir_: Path):
    mp = defaultdict(list)
    for txt in dir_.glob("*.txt"):
        img = txt.stem
        for l in txt.read_text().splitlines():
            if not l.strip():
                continue
            cls, *bbox = map(float, l.split()[:5])
            mp[img].append({"category_id": int(cls), "bbox": bbox})
    return mp

# ─────────────────── Origins CSV ────────────────

def load_origins(csv_path: Path):
    """Load semicolon-separated CSV into dict {image_id: (Ox,Oy,Oz)}."""
    if not csv_path.exists():
        print("⚠️  No SourcePoints.csv – all origins at (0,0,0)")
        return {}
    origins = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=';')
        header = [h.strip().lower() for h in next(reader)]
        try:
            idx_id = header.index('imageid')
            idx_x = header.index('x')
            idx_y = header.index('y')
            idx_z = header.index('z')
        except ValueError:
            raise RuntimeError("SourcePoints.csv must have columns: ImageId;X;Y;Z")
        for row in reader:
            if not row or len(row) < max(idx_z, idx_y, idx_x) + 1:
                continue
            img_id = row[idx_id].strip()
            try:
                Ox = float(row[idx_x].replace(',', '.'))
                Oy = float(row[idx_y].replace(',', '.'))
                Oz = float(row[idx_z].replace(',', '.'))
            except ValueError:
                print(f"⚠️  Bad numeric values in {csv_path} line: {row}")
                continue
            origins[img_id] = (Ox, Oy, Oz)
    return origins

# ─────────────── Overlay + XYZ ─────────────────

def overlay_xyz(rgb, dets, depth, origin, radioOrigin):
    Xorigin, Yorigin, Zorigin = origin
    XradioOrigin, YradioOrigin, ZradioOrigin = radioOrigin
    H, W = rgb.shape[:2]
    vis = rgb.copy(); 
    out = []
    
    for det_id, det in enumerate(dets, 1):
        x0, y0, x1, y1 = bbox_px(det['bbox'], W, H)
        if x1 <= x0 or y1 <= y0:
            continue
        patch = depth[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        
        h, w = patch.shape; 
        dy, dx = max(1, int(0.05 * h / 2)), max(1, int(0.05 * w / 2))
        cy, cx = h // 2, w // 2    
        d_val = float(np.nanmedian(patch[cy - dy:cy + dy + 1, cx - dx:cx + dx + 1]))
        
        # --- kierunek 3-D ---
        u_c, v_c = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        phi_deg, theta_deg = equiXYToPhiThetaInDeg(u_c, v_c, W, H)
        dir_vec            = equiXYToDirectionVector(u_c, v_c, W, H)
        Xunit, Yunit, Zunit         = map(float, dir_vec)               # jednostkowy
        Xscaled, Yscaled, Zscaled         = (c * d_val for c in dir_vec)      # skalowany
        Xglobal, Yglobal, Zglobal         = Xscaled + Xorigin, Yscaled + Yorigin, Zscaled + Zorigin         # world
        Xradio, Yradio, Zradio            = Xscaled + XradioOrigin, Yscaled + YradioOrigin, Zscaled + ZradioOrigin # radio-based localization
        
        # --- wizualizacja ---
        label = CLASSES.get(det['category_id'], f"id{det['category_id']}")
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 4)
        
        depth_txt  = f"d={d_val:.2f} m"
        angle_txt  = f"phi={phi_deg:+.1f} deg   theta={theta_deg:+.1f} deg"
        unit_txt   = f"unit=({Xunit:+.3f}, {Yunit:+.3f}, {Zunit:+.3f})"
        scaled_txt  = f"scaled =({Xscaled:+.2f}, {Yscaled:+.2f}, {Zscaled:+.2f})"
        global_txt = f"global =({Xglobal:+.2f}, {Yglobal:+.2f}, {Zglobal:+.2f})"
        radio_txt = f"radio =({Xradio:+.2f}, {Yradio:+.2f}, {Zradio:+.2f})"
        
        # base_x, base_y = x0, max(0, y0 - 5)    # nad ramką
        # font, scale, thick, dy = cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2, 24
        # for i, txt in enumerate((label, depth_txt, angle_txt, unit_txt, scaled_txt, global_txt, radio_txt)):
        #     cv2.putText(vis, txt, (base_x, base_y - (len(txt.splitlines())-i)*dy),
        #                 font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        text     = f"#{det_id}"
        org      = (x0 + 6, y0 + 45)            # pozycja napisu
        font     = cv2.FONT_HERSHEY_DUPLEX
        scale    = 2.2

        # czarna obwódka (grubsza)
        cv2.putText(vis, text, org, font, scale,
                    (0, 0, 0), 7, cv2.LINE_AA)  # grubość 7 – kontur

        # biały napis wewnątrz
        cv2.putText(vis, text, org, font, scale,
                    (0, 215, 255), 2, cv2.LINE_AA)
            
        out.append((det_id, label, d_val, Xglobal, Yglobal, Zglobal, phi_deg, theta_deg, Xradio, Yradio, Zradio))
    
    return vis, out

# ────────────────── Main ────────────────────────

def main(a):
    global cursorPos

    root = Path(a.data_dir)
    imgs = root / "images"; met = root / "depth/metric"; rel = root / "depth/relative"; lbl = root / "labels"
    out_dir = root / "out/xyz"; out_dir.mkdir(parents=True, exist_ok=True)

    origins = load_origins(root / "SourcePoints.csv")
    radioOrigins = load_origins(root / "RadioPoints.csv")

    preds = load_yolo(lbl)
    if not preds:
        print("No label txts"); sys.exit(1)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, mouseCallback)  # ← aktywujemy śledzenie kursora

    disp_w = min(a.max_w, IMG_W)
    disp_h = min(a.max_h, IMG_H)

    for img_id, dets in sorted(preds.items()):
        img_fp = next((imgs / f"{img_id}.{e}" for e in ("jpg", "png") if (imgs / f"{img_id}.{e}").exists()), None)
        if img_fp is None:
            print("no img", img_id); continue
        rgb = cv2.imread(str(img_fp))
        depth = None; metric = True
        if a.mode == 'metric':
            dp = met / f"{img_id}_raw_depth_meter.npy"
            if dp.exists():
                depth = load_metric(dp)
        else:
            dpng, dtxt = rel / f"{img_id}_depth.png", rel / f"{img_id}.txt"
            if dpng.exists() and dtxt.exists():
                depth = load_relative(dpng, dtxt, DEF_REF_DIST); metric = False
        if depth is None:
            print("no depth", img_id); continue

        vis, xyz = overlay_xyz(rgb, dets, depth, origins.get(img_id, (0.0, 0.0, 0.0)), radioOrigins.get(img_id, (0.0,0.0,0.0)))

        # ───────── statyczny podpis obrazka (ImageId) ─────────
        img_text = f"{img_id}"
        org      = (12, 50)                     # 12 px od lewej, 50 px od góry
        font     = cv2.FONT_HERSHEY_DUPLEX
        scale    = 2.4

        # kontur (czarny)
        cv2.putText(vis, img_text, org, font, scale,
                    (0, 0, 0), 7, cv2.LINE_AA)

        # wypełnienie (czerw)
        cv2.putText(vis, img_text, org, font, scale,
                    (255, 0, 0), 3, cv2.LINE_AA)

        Xorigin, Yorigin, Zorigin = origins.get(img_id, (0.0, 0.0, 0.0))
        XradioOrigin, YradioOrigin, ZradioOrigin = radioOrigins.get(img_id, (0.0,0.0,0.0))
        
        rows = []
        for det_id, lbl, d, Xglobal, Yglobal, Zglobal, phi, theta, Xradio, Yradio, Zradio in xyz:
            rows.append([
                det_id,
                lbl,
                f"{d:.2f}",
                f"{phi:+.1f}",
                f"{theta:+.1f}",
                f"{Xglobal:.2f}", f"{Yglobal:.2f}", f"{Zglobal:.2f}",
                f"{Xradio:.2f}", f"{Yradio:.2f}", f"{Zradio:.2f}",
            ])

        csv_path = out_dir / f"{img_id}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=';')
            # nagłówek
            writer.writerow([
                "DetId", "Label", "Distance", "Phi", "Theta",
                "Xglobal", "Yglobal", "Zglobal",
                "Xradio",  "Yradio",  "Zradio"
            ])
            # dane
            writer.writerows(rows)

        # ———————————— Podgląd interaktywny ————————————
        cv2.resizeWindow(WIN, disp_w, disp_h)
        while True:
            display = vis.copy()
            if cursorPos is not None:
                u, v = cursorPos
                if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                    d_val = float(depth[v, u])
                    
                    if np.isfinite(d_val):
                        # red dot
                        cv2.circle(display, (u, v), 8, (0, 0, 255), -1)

                        # math
                        phi_deg, theta_deg = equiXYToPhiThetaInDeg(u, v, rgb.shape[1], rgb.shape[0])
                        dir_vec            = equiXYToDirectionVector(u, v, rgb.shape[1], rgb.shape[0])
                        Xunit, Yunit, Zunit         = map(float, dir_vec)                   # jednostkowy
                        Xscaled, Yscaled, Zscaled         = (c * d_val for c in dir_vec)          # skalowany głębokością
                        Xglobal, Yglobal, Zglobal         = Xscaled + Xorigin, Yscaled + Yorigin, Zscaled + Zorigin             # + przesunięcie
                        Xradio, Yradio, Zradio            = Xscaled + XradioOrigin, Yscaled + YradioOrigin, Zscaled + ZradioOrigin # radio-based localization

                        # texts
                        depth_txt  = f"d={d_val:.2f} m"
                        angle_txt  = f"phi={phi_deg:+.1f} deg   theta={theta_deg:+.1f} deg"
                        unit_txt   = f"unit=({Xunit:+.3f}, {Yunit:+.3f}, {Zunit:+.3f})"
                        scaled_txt  = f"scaled =({Xscaled:+.2f}, {Yscaled:+.2f}, {Zscaled:+.2f})"
                        global_txt = f"world =({Xglobal:+.2f}, {Yglobal:+.2f}, {Zglobal:+.2f})"
                        radio_txt = f"radio =({Xradio:+.2f}, {Yradio:+.2f}, {Zradio:+.2f})"

                        # style
                        base_x, base_y = u + 12, v - 12      # lewy-górny narożnik napisu
                        font      = cv2.FONT_HERSHEY_SIMPLEX
                        scale     = 1.1                      # większa czcionka
                        thickness = 3
                        dy        = int(32 * scale)

                        # wiersz 1 – głębokość
                        cv2.putText(display, depth_txt,  (base_x, base_y + 0*dy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

                        # wiersz 2 – kąty (odsuwamy o ~35 pikseli w dół)
                        cv2.putText(display, angle_txt,  (base_x, base_y + 1*dy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)
                        
                        cv2.putText(display, unit_txt, (base_x, base_y + 2*dy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

                        cv2.putText(display, scaled_txt, (base_x, base_y + 3*dy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

                        cv2.putText(display, global_txt, (base_x, base_y + 4*dy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

                        cv2.putText(display, radio_txt, (base_x, base_y + 5*dy), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

            cv2.imshow(WIN, display)
            k = cv2.waitKey(20) & 0xFF
            if k == ord(' '):
                break  # następny obraz
            if k in (ord('q'), 27):
                cv2.destroyAllWindows(); return

    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data')
    ap.add_argument('--mode', choices=['metric', 'relative'], default='metric')
    ap.add_argument('--max_w', type=int, default=DEF_MAX_W)
    ap.add_argument('--max_h', type=int, default=DEF_MAX_H)
    main(ap.parse_args())
