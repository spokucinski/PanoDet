import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics

from lib.GtEntry import GtEntry
from lib.VisualDetection import VisualDetection
from lib.IJDCalculationEntry import CalculationEntry
from lib.Visualizer import Visualizer
from lib.RadioDetection import RadioPrediction

from lib.Validator import Validator
from lib.DataLoader import DataLoader

from typing import List
from consts.ObjectClasses import CODE55_CLASSES

def compute_euclidean_errors(
    detections: List[VisualDetection], 
    gt_entries: List[GtEntry]
) -> None:
    gt_map = {entry.objectId: entry for entry in gt_entries}
    errors_global = []
    errors_radio = []
    skipped = 0

    for det in detections:
        if not det.detectionCorrect:
            continue
        gt = gt_map.get(det.detectedObjectId)
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

def main(
        groundTruthPath: str,
        visualDetectionsPath: str
    ) -> None:

    gtEntries = DataLoader.readGroundTruth(groundTruthPath)
    visualDetections = DataLoader.readVisionDetections(visualDetectionsPath)
    compute_euclidean_errors(visualDetections, gtEntries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyzer of visual detections.")
    
    # Base parameters
    parser.add_argument("--gtPath", type=str, default="data/GroundTruth.csv", help="File path to GroundTruth.csv")
    parser.add_argument("--visualDetectionsPath", type=str, default="data/DetectionsSummary.csv", help="File path to DetectionsSummary.csv")

    args = parser.parse_args()
    
    main(groundTruthPath=args.gtPath, 
         visualDetectionsPath=args.visualDetectionsPath,)