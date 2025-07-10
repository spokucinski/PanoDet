import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics

from lib.GtEntry import GtEntry
from lib.Detection import Detection
from lib.IJDCalculationEntry import IJDCalculationEntry
from lib.Visualizer import Visualizer
from lib.RadioDetection import RadioDetection

from lib.Validator import Validator
from lib.DataLoader import DataLoader

from typing import List
from consts.ObjectClasses import CODE55_CLASSES

def compute_euclidean_errors(
    detections: List[Detection], 
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

def calculateWZO(predictedClass: str, expectedClasses: set) -> int:
    """
    Calculates the Expectation Compliance Coefficient (WZO).
    Returns 1 if the object's class is in the expected classes set, otherwise 0.
    """
    return 1 if predictedClass in expectedClasses else 0

def calculateVisualDetectionCorrectness(
    gtObjectId: str,
    visualDetections: List[Detection]
) -> int:
    """
    Returns 1 if there is a correct visual detection for this object, else 0.
    """
    for visualDetection in visualDetections:
        if visualDetection.detectedObjectId == gtObjectId and visualDetection.detectionCorrect is True:
            return 1
    return 0

def calculateRadioAvailability(
    gtObjectId: str,
    radioPredictions: List[RadioDetection]
) -> int:
    """
    Returns 1 if there is at least one radio prediction for this object, else 0.
    """
    for radioPrediction in radioPredictions:
        if radioPrediction.objectId == gtObjectId:
            return 1
    return 0

def calculateMeanVisualError(gtEntry: GtEntry, visualDetections: List[Detection]) -> float | None:
    """
    Calculates the mean Euclidean error for all correct visual detections matching the gtEntry.
    Returns 0 if no detections are found.
    """
    matchingDetections = [
        d for d in visualDetections
        if d.detectedObjectId == gtEntry.objectId and d.detectionCorrect is True
    ]
    if not matchingDetections:
        return 0
    errors = []
    for d in matchingDetections:
        dx = gtEntry.xgt - d.xglobal
        dy = gtEntry.ygt - d.yglobal
        dz = gtEntry.zgt - d.zglobal
        error = math.sqrt(dx*dx + dy*dy + dz*dz)
        errors.append(error)
    return float(sum(errors) / len(errors))

def calculateMeanRadioError(gtEntry: GtEntry, radioPredictions: List[RadioDetection]) -> float | None:
    """
    Calculates the mean Euclidean error for all radio predictions matching the gtEntry.
    Returns 0 if no predictions are found.
    """
    matchingRadio = [
        r for r in radioPredictions
        if r.objectId == gtEntry.objectId
    ]
    if not matchingRadio:
        return 0
    errors = []
    for r in matchingRadio:
        dx = gtEntry.xgt - r.xr
        dy = gtEntry.ygt - r.yr
        dz = gtEntry.zgt - r.zr
        error = math.sqrt(dx*dx + dy*dy + dz*dz)
        errors.append(error)
    return float(sum(errors) / len(errors))

def calculateWLR(e_rad: float | None, k_rad: float) -> float | None:
    if e_rad is None:
        return 0.0
    return math.exp(-k_rad * e_rad)

def calculateWLW(e_wiz: float | None, k_wiz: float) -> float | None:
    if e_wiz is None:
        return 0.0
    return math.exp(-k_wiz * e_wiz)

def aggregateIJDStatistics(ijdResults: List[IJDCalculationEntry]) -> None:
    ijd_values = [entry.ijd for entry in ijdResults]
    print(f"\n===== IJD AGGREGATION =====")
    if ijd_values:
        mean_val = sum(ijd_values) / len(ijd_values)
        min_val = min(ijd_values)
        max_val = max(ijd_values)
        median_val = statistics.median(ijd_values)
        std_val = statistics.stdev(ijd_values) if len(ijd_values) > 1 else 0.0
        print(f"Apartment IJD mean:   {mean_val:.4f}")
        print(f"Apartment IJD median: {median_val:.4f}")
        print(f"Apartment IJD min:    {min_val:.4f}")
        print(f"Apartment IJD max:    {max_val:.4f}")
        print(f"Apartment IJD std:    {std_val:.4f}")
    else:
        print("No IJD results to aggregate.")

    # Per room stats
    room2ijds = {}
    for entry in ijdResults:
        room = entry.gtEntry.room
        if room not in room2ijds:
            room2ijds[room] = []
        room2ijds[room].append(entry.ijd)

    print("\nMean IJD per room:")
    for room, ijds in room2ijds.items():
        if ijds:
            mean_val = sum(ijds) / len(ijds)
            min_val = min(ijds)
            max_val = max(ijds)
            median_val = statistics.median(ijds)
            std_val = statistics.stdev(ijds) if len(ijds) > 1 else 0.0
            print(f"  {room:20s} | mean: {mean_val:.4f} | median: {median_val:.4f} | min: {min_val:.4f} | max: {max_val:.4f} | std: {std_val:.4f} (n={len(ijds)})")
        else:
            print(f"  {room:20s} | No data")

def computeIJD(
        gtEntries: List[GtEntry], 
        visualDetections: List[Detection], 
        radioPredictions: List[RadioDetection],
        radioWeight: float = 0.5,
        visionWeight: float = 0.5,
        radioSensitivity: float = 0.5,
        visionSensitivity: float = 0.5
    ) -> None:
    
    # Watch out for decimal reprezentation
    assert abs((radioWeight + visionWeight) - 1.0) < 1e-6, "Weights must sum to 1.0"

    ijdResults = []

    for entry in gtEntries:
        wzo = calculateWZO(entry.code55Class, CODE55_CLASSES)
        cWiz = calculateVisualDetectionCorrectness(entry.objectId, visualDetections)
        cRad = calculateRadioAvailability(entry.objectId, radioPredictions)
        
        eWiz = calculateMeanVisualError(entry, visualDetections)
        eRad = calculateMeanRadioError(entry, radioPredictions)
    
        wlw = calculateWLW(eWiz, visionSensitivity)
        wlr = calculateWLR(eRad, radioSensitivity)
        
        ijd = wzo * (radioWeight * cRad * wlr + visionWeight * cWiz * wlw)

        ijdEntry = IJDCalculationEntry(
            gtEntry=entry, 
            wzo=wzo,
            cwiz=cWiz,
            crad=cRad,
            visionWeight=visionWeight,
            radioWeight=radioWeight,
            visionError=eWiz,
            radioError=eRad,
            kRad=radioSensitivity,
            kWiz=visionSensitivity,
            wlw=wlw,
            wlr=wlr,
            ijd=ijd)
        
        ijdResults.append(ijdEntry)

        print(
            f"ObjectId: {entry.objectId:20s} | "
            f"CODE55: {entry.code55Class:20s} | "
            f"WZO: {wzo} | "
            f"C_WIZ: {cWiz} | "
            f"C_RAD: {cRad} | "
            f"visionWeight: {visionWeight:.2f} | "
            f"radioWeight: {radioWeight:.2f} | "
            f"visionError: {eWiz:.2f} | "
            f"radioError: {eRad:.2f} | "
            f"WLW: {wlw:.2f} | "
            f"WLR: {wlr:.2f} | "
            f"IJD: {ijd:.2f}"
        )
    
    aggregateIJDStatistics(ijdResults)


def main(groundTruthPath: str, 
         visualDetectionsPath: str, 
         radioPredictionsPath: str,
         radioWeight: float = 0.5,
         visionWeight: float = 0.5,
         radioSensitivity: float = 0.5,
         visionSensitivity: float = 0.5) -> None:
    
    gtEntries = DataLoader.readGroundTruth(groundTruthPath)
    Validator.validateGtClasses(gtEntries)

    Visualizer.printGtStatistics(gtEntries)
    Visualizer.saveGtStatisticsHistograms(gtEntries)

    visualDetections = DataLoader.readVisualDetections(visualDetectionsPath)
    Validator.validateDetectionLabels(visualDetections)
    Validator.validateDetectedObjectIds(visualDetections, gtEntries)

    Visualizer.printVisualDetectionsStatistics(visualDetections)
    Visualizer.saveVisualDetectionsHistograms(visualDetections)

    radioPredictions = DataLoader.readRadioPredictions(radioPredictionsPath) 
    Validator.validateRadioObjectIds(radioPredictions, gtEntries)
    
    computeIJD(gtEntries=gtEntries, 
               visualDetections=visualDetections, 
               radioPredictions=radioPredictions,
               radioWeight=radioWeight,
               visionWeight=visionWeight,
               radioSensitivity=radioSensitivity,
               visionSensitivity=visionSensitivity)

    compute_euclidean_errors(visualDetections, gtEntries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculation script for the IJD.")
    
    # Base parameters
    parser.add_argument("--gtPath", type=str, default="data/GroundTruth.csv", help="File path to GroundTruth.csv")
    parser.add_argument("--visualDetectionsPath", type=str, default="data/DetectionsSummary.csv", help="File path to DetectionsSummary.csv")
    parser.add_argument("--radioPredictionsPath", type=str, default="data/RadioDetections.csv", help="File path to RadioDetections.csv")
    
    # Calculation parameters
    parser.add_argument("--radioWeight", type=float, default=0.5, help="Weight for radio subsystem (default: 0.5)")
    parser.add_argument("--visionWeight", type=float, default=0.5, help="Weight for vision subsystem (default: 0.5)")
    parser.add_argument("--radioSensitivity", type=float, default=0.5, help="Error sensitivity for radio (k_RAD, default: 0.5)")
    parser.add_argument("--visionSensitivity", type=float, default=0.5, help="Error sensitivity for vision (k_WIZ, default: 0.5)")

    args = parser.parse_args()
    
    main(groundTruthPath=args.gtPath, 
         visualDetectionsPath=args.visualDetectionsPath, 
         radioPredictionsPath=args.radioPredictionsPath,
         radioWeight=args.radioWeight,
         visionWeight=args.visionWeight,
         radioSensitivity=args.radioSensitivity,
         visionSensitivity=args.visionSensitivity)