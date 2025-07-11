import argparse
import numpy as np
import matplotlib.pyplot as plt
import argparse

from lib.GtEntry import GtEntry
from lib.VisualDetection import VisualDetection
from lib.Visualizer import Visualizer
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

def main(gtPath, visualDetectionsPath, skipValidation=False):

    gtEntries = DataLoader.readGroundTruth(gtPath)
    visualDetections = DataLoader.readVisualDetections(visualDetectionsPath)

    if not skipValidation:
        Validator.validateVisualDetectionLabels(visualDetections, CODE55_CLASSES)
        Validator.validateVisualDetectionObjectIds(visualDetections, gtEntries)

    outDir = "data/out/vision"
    Visualizer.saveDetectionsPerRoomHistogram(visualDetections=visualDetections, outDir=outDir)
    Visualizer.saveDetectionsPerRoomHistogram(visualDetections=visualDetections, outDir=outDir, polish=True)

    Visualizer.saveCorrectIncorrectPerRoomHistogram(visualDetections=visualDetections,
                                                    outDir=outDir,
                                                    polish=False)
    
    Visualizer.saveCorrectIncorrectPerRoomHistogram(visualDetections=visualDetections,
                                                    outDir=outDir,
                                                    polish=True)
    
    Visualizer.saveDetectionsPerLabelHistogram(visualDetections=visualDetections,
                                               outDir=outDir,
                                               topN=15,
                                               polish=False)
    
    Visualizer.saveDetectionsPerLabelHistogram(visualDetections=visualDetections,
                                               outDir=outDir,
                                               topN=15,
                                               polish=True)
    
    Visualizer.saveDetectionCorrectnessPerClassBar(visualDetections=visualDetections,
                                                   outDir=outDir,
                                                   topN=15,
                                                   polish=False)
    
    Visualizer.saveDetectionCorrectnessPerClassBar(visualDetections=visualDetections,
                                                   outDir=outDir,
                                                   topN=15,
                                                   polish=True)
    
    Visualizer.saveDetectionDistancesHistogram(visualDetections=visualDetections,
                                               outDir=outDir,
                                               polish=False,
                                               correctOnly=False)
    
    Visualizer.saveDetectionDistancesHistogram(visualDetections=visualDetections,
                                               outDir=outDir,
                                               polish=True,
                                               correctOnly=False)
    
    Visualizer.saveDetectionDistancesHistogram(visualDetections=visualDetections,
                                               outDir=outDir,
                                               polish=False,
                                               correctOnly=True)
    
    Visualizer.saveDetectionDistancesHistogram(visualDetections=visualDetections,
                                               outDir=outDir,
                                               polish=True,
                                               correctOnly=True)
    
    Visualizer.saveDetectionCorrectnessPie(visualDetections=visualDetections,
                                           outDir=outDir,
                                           polish=False)
    
    Visualizer.saveDetectionCorrectnessPie(visualDetections=visualDetections,
                                           outDir=outDir,
                                           polish=True)
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analyzer for performance of the visual subsystem only.")

    parser.add_argument("--gtPath", type=str, default="data/GroundTruth.csv", help="CSV file path for Ground Truth")
    parser.add_argument("--visualDetectionsPath", type=str, default="data/DetectionsSummary.csv", help="CSV file path for Visual Detections")
    parser.add_argument("--skipValidation", action="store_true", default=False, help="Skip input validation (default: False)")
    
    args = parser.parse_args()

    main(
        gtPath=args.gtPath,
        visualDetectionsPath=args.visualDetectionsPath,
        skipValidation=args.skipValidation
    )
