import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GtEntry import GtEntry
from Detection import Detection
from RadioDetection import RadioDetection
from typing import Tuple, List
from ObjectClasses import OBJECT_CLASSES


def str_to_bool_with_check(val) -> Tuple[bool, str]:
    """Zwraca (wynik, status): wynik bool, status: 'TRUE', 'FALSE', 'ERROR'"""
    if isinstance(val, str):
        val_str = val.strip().upper()
        if val_str == "TRUE":
            return True, "TRUE"
        if val_str == "FALSE":
            return False, "FALSE"
    return False, "ERROR"

def validate_gt_classes(gt_entries: List[GtEntry]) -> None:
    not_found = []
    for entry in gt_entries:
        if entry.code55 not in OBJECT_CLASSES:
            not_found.append((entry.object_id, entry.code55))
    if not_found:
        print("Nierozpoznane klasy CODE55 w Ground Truth:")
        for obj_id, code in not_found:
            print(f"  ObjectId: {obj_id}  |  CODE55: '{code}'")
    else:
        print("Wszystkie klasy CODE55 w GT są prawidłowe.")

def validate_detection_labels(detections: List[Detection]) -> None:
    not_found = []
    for det in detections:
        if det.label not in OBJECT_CLASSES:
            not_found.append((det.rel_det_number, det.label))
    if not_found:
        print("Nierozpoznane klasy Label w Detekcjach:")
        for rel_num, label in not_found:
            print(f"  RelDetNumber: {rel_num}  |  Label: '{label}'")
    else:
        print("Wszystkie klasy Label w Detekcjach są prawidłowe.")

def validate_detected_object_ids(
    detections: List[Detection],
    gt_entries: List[GtEntry]
) -> None:
    gt_object_ids = {entry.object_id for entry in gt_entries}
    matched = 0
    unmatched = 0
    unmatched_list = []
    for det in detections:
        if det.detected_object_id in gt_object_ids:
            matched += 1
        else:
            unmatched += 1
            unmatched_list.append((det.rel_det_number, det.detected_object_id))
    print(f"Liczba detekcji z dopasowanym DetectedObjectId: {matched}")
    print(f"Liczba detekcji bez dopasowania DetectedObjectId: {unmatched}")
    if unmatched > 0:
        print("Przykłady detekcji bez dopasowania:")
        for rel_num, obj_id in unmatched_list[:10]:  # np. pokaż tylko 10 pierwszych
            print(f"  RelDetNumber: {rel_num}  |  DetectedObjectId: '{obj_id}'")

def read_gt_csv(path: str) -> List[GtEntry]:
    df = pd.read_csv(path, sep=';')
    entries = [
        GtEntry(
            ground_truth_id=int(row['GroundTruthId']),
            room=row['Room'],
            collection=row['Collection'],
            object_id=row['ObjectId'],
            code55=row['CODE55'],
            xgt=float(row['Xgt']),
            ygt=float(row['Ygt']),
            zgt=float(row['Zgt'])
        )
        for _, row in df.iterrows()
    ]
    return entries

def read_radio_detections_csv(path: str) -> List[RadioDetection]:
    import pandas as pd
    df = pd.read_csv(path, sep=';')
    entries = [
        RadioDetection(
            radio_id=int(row['RadioId']),
            experiment=row['Experiment'],
            room=row['Room'],
            tracker_id=row['TrackerId'],
            object_id=row['ObjectId'],
            xr=float(row['Xr']),
            yr=float(row['Yr']),
            zr=float(row['Zr']),
            xrgt=float(row['Xrgt']),
            yrgt=float(row['Yrgt']),
            zrgt=float(row['Zrgt'])
        )
        for _, row in df.iterrows()
    ]
    return entries

def read_detections_csv(path: str) -> Tuple[List[Detection], int, int, int, List[int]]:
    import pandas as pd
    df = pd.read_csv(path, sep=';')
    detections = []
    true_count = 0
    false_count = 0
    error_count = 0
    error_rows = []

    for idx, row in df.iterrows():
        bool_val, status = str_to_bool_with_check(row['DetectionCorrect'])
        if status == "TRUE":
            true_count += 1
        elif status == "FALSE":
            false_count += 1
        else:
            error_count += 1
            error_rows.append(idx + 2)  # +2 bo pandas liczy od 0, a 1 to nagłówek w CSV

        detection = Detection(
            detection_id=int(row['DetectionId']),
            rel_det_number=int(row['RelDetNumber']),
            label=row['Label'],
            distance=float(row['Distance']),
            phi=float(row['Phi']),
            theta=float(row['Theta']),
            xglobal=float(row['Xglobal']),
            yglobal=float(row['Yglobal']),
            zglobal=float(row['Zglobal']),
            xradio=float(row['Xradio']),
            yradio=float(row['Yradio']),
            zradio=float(row['Zradio']),
            room=row['Room'],
            image_id=row['ImageId'],
            detected_object_id=row['DetectedObjectId'],
            detection_correct=bool_val,
            detected_in_room=row['DetectedInRoom']
        )
        detections.append(detection)
    return detections, true_count, false_count, error_count, error_rows

def validate_radio_object_ids(
    radio_detections: List[RadioDetection],
    gt_entries: List[GtEntry]
) -> None:
    """
    Sprawdza, czy dla każdej detekcji radiowej pole object_id znajduje się w GroundTruth.
    Wypisuje liczby dopasowań i niedopasowań oraz przykłady brakujących.
    """
    gt_object_ids = {entry.object_id for entry in gt_entries}
    matched = 0
    unmatched = 0
    unmatched_list = []
    for radio in radio_detections:
        if radio.object_id in gt_object_ids:
            matched += 1
        else:
            unmatched += 1
            unmatched_list.append((radio.radio_id, radio.object_id))
    print(f"Liczba radio detekcji z dopasowanym ObjectId: {matched}")
    print(f"Liczba radio detekcji bez dopasowania ObjectId: {unmatched}")
    if unmatched > 0:
        print("Przykłady radio detekcji bez dopasowania:")
        for radio_id, obj_id in unmatched_list[:10]:  # wyświetl tylko 10 pierwszych
            print(f"  RadioId: {radio_id}  |  ObjectId: '{obj_id}'")

def compute_euclidean_errors(
    detections: List[Detection], 
    gt_entries: List[GtEntry]
) -> None:
    gt_map = {entry.object_id: entry for entry in gt_entries}
    errors_global = []
    errors_radio = []
    skipped = 0

    for det in detections:
        if not det.detection_correct:
            continue
        gt = gt_map.get(det.detected_object_id)
        if gt is None:
            skipped += 1
            continue
        err_global = np.linalg.norm([
            det.xglobal - gt.xgt,
            det.yglobal - gt.ygt,
            det.zglobal - gt.zgt
        ])
        err_radio = np.linalg.norm([
            det.xradio - gt.xgt,
            det.yradio - gt.ygt,
            det.zradio - gt.zgt
        ])
        errors_global.append(err_global)
        errors_radio.append(err_radio)

    def stats(arr):
        arr = np.array(arr)
        return {
            "mean": arr.mean() if len(arr) > 0 else None,
            "median": np.median(arr) if len(arr) > 0 else None,
            "min": arr.min() if len(arr) > 0 else None,
            "max": arr.max() if len(arr) > 0 else None,
            "std": arr.std(ddof=1) if len(arr) > 1 else None,
            "count": len(arr)
        }

    stats_g = stats(errors_global)
    stats_r = stats(errors_radio)

    print("\nSTATYSTYKI BŁĘDU EUKLIDESOWEGO (Tylko poprawne detekcje z dopasowaniem):")
    print("\n-> Błąd względem Xglobal/Yglobal/Zglobal:")
    for k, v in stats_g.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\n-> Błąd względem Xradio/Yradio/Zradio:")
    for k, v in stats_r.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nLiczba pominiętych detekcji (brak dopasowania do GT): {skipped}")

    # HISTOGRAMY
    if len(errors_global) > 0:
        plt.figure(figsize=(9,4))
        plt.subplot(1,2,1)
        plt.hist(errors_global, bins=25, edgecolor='black')
        plt.title('Histogram błędów euklidesowych (Global)')
        plt.xlabel('Błąd [m]')
        plt.ylabel('Liczba detekcji')
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.hist(errors_radio, bins=25, edgecolor='black', color='orange')
        plt.title('Histogram błędów euklidesowych (Radio)')
        plt.xlabel('Błąd [m]')
        plt.ylabel('Liczba detekcji')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

def main(gt_path: str, detections_path: str, radio_path: str) -> None:
    gt_entries = read_gt_csv(gt_path)
    validate_gt_classes(gt_entries)

    detections, true_count, false_count, error_count, error_rows = read_detections_csv(detections_path)
    validate_detection_labels(detections)

    validate_detected_object_ids(detections, gt_entries)

    radio_detections = read_radio_detections_csv(radio_path)
    print(f"Liczba detekcji radiowych: {len(radio_detections)}")
    print("Przykładowe detekcje radiowe:", radio_detections[:2])

    validate_radio_object_ids(radio_detections, gt_entries)

    print(f"Liczba wpisów ground truth: {len(gt_entries)}")
    print(f"Liczba detekcji: {len(detections)}")
    print(f"Liczba poprawnych detekcji (TRUE): {true_count}")
    print(f"Liczba niepoprawnych detekcji (FALSE): {false_count}")
    print(f"Liczba nieprawidłowych wartości DetectionCorrect: {error_count}")
    if error_count > 0:
        print(f"Wiersze z błędem: {error_rows}")
    print("Przykładowe obiekty Detection:", detections[:2])

    compute_euclidean_errors(detections, gt_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Obróbka wyników GT i detekcji.")
    parser.add_argument("--gt_path", type=str, default="data/GroundTruth.csv", help="Ścieżka do pliku GroundTruth.csv")
    parser.add_argument("--detections_path", type=str, default="data/DetectionsSummary.csv", help="Ścieżka do pliku DetectionsSummary.csv")
    parser.add_argument("--radio_path", type=str, default="data/RadioDetections.csv", help="Ścieżka do pliku RadioDetections.csv")
    args = parser.parse_args()
    main(gt_path=args.gt_path, detections_path=args.detections_path, radio_path=args.radio_path)

