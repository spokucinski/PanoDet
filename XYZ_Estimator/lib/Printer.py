import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List
from lib.GtEntry import GtEntry
from consts.ObjectClasses import CODE55_CLASSES
from lib.VisualDetection import VisualDetection
import numpy as np

class Printer:

    @staticmethod
    def printGtStatistics(gtEntries: List[GtEntry]) -> None:
        print("########## Ground Truth Statistics ##########")
        
        total = len(gtEntries)
        print(f"Total number of ground truth samples: {total}")

        # Count objects per room
        countPerRoom = Counter(entry.room for entry in gtEntries)
        print("\nObject count per room:")
        for room, count in countPerRoom.items():
            print(f"  {room:20s} : {count}")

        # Count objects per class label
        countPerClass = Counter(entry.classLabel for entry in gtEntries)
        print("\nObject count per class label:")
        for label, count in countPerClass.items():
            print(f"  {label:20s} : {count}")

        # Most common class label and room
        if countPerClass:
            mostCommonClass = countPerClass.most_common(1)[0]
            print(f"\nMost common class label: '{mostCommonClass[0]}' ({mostCommonClass[1]} occurrences)")
        else:
            print("\nNo class labels found.")

        if countPerRoom:
            mostCommonRoom = countPerRoom.most_common(1)[0]
            print(f"Most common room: '{mostCommonRoom[0]}' ({mostCommonRoom[1]} occurrences)")
        else:
            print("No rooms found.")
            
    @staticmethod
    def printVisualDetectionsStatistics(visualDetections: List[VisualDetection]) -> None:
        print("########## Visual Detections Statistics ##########")
        
        total = len(visualDetections)
        print(f"Total number of visual detections: {total}")

        correctlyRecognizedCount = sum(1 for detection in visualDetections if detection.detectionCorrect is True)
        wronglyRecognizedCount = sum(1 for detection in visualDetections if detection.detectionCorrect is False)

        print(f"Number of correct detections (TRUE): {correctlyRecognizedCount}")
        print(f"Number of incorrect detections (FALSE): {wronglyRecognizedCount}")

        if total != correctlyRecognizedCount + wronglyRecognizedCount:
            print(f"Number of detections with invalid DetectionCorrect value: {total - (correctlyRecognizedCount + wronglyRecognizedCount)}")

        # Detections per room
        countPerRoom = Counter(d.room for d in visualDetections)
        print("\nDetections per room:")
        for room, count in countPerRoom.items():
            print(f"  {room:20s} : {count}")

        # Detections per label/class
        countPerLabel = Counter(detection.label for detection in visualDetections)
        print("\nDetections per label/class:")
        for label, count in countPerLabel.items():
            print(f"  {label:20s} : {count}")

        # Correct/incorrect ratio per class
        correctPerLabel = defaultdict(int)
        incorrectPerLabel = defaultdict(int)
        for d in visualDetections:
            if d.detectionCorrect is True:
                correctPerLabel[d.label] += 1
            elif d.detectionCorrect is False:
                incorrectPerLabel[d.label] += 1

        print("\nCorrect/Incorrect detections per label/class:")
        allLabels = sorted(set(correctPerLabel) | set(incorrectPerLabel))
        for label in allLabels:
            c = correctPerLabel.get(label, 0)
            w = incorrectPerLabel.get(label, 0)
            ratio = f"{c}/{c+w}" if (c+w) > 0 else "n/a"
            print(f"  {label:20s} : {c} correct, {w} incorrect   (accuracy: {ratio})")

        # Correct/incorrect ratio per room
        correctPerRoom = defaultdict(int)
        incorrectPerRoom = defaultdict(int)
        for d in visualDetections:
            if d.detectionCorrect is True:
                correctPerRoom[d.room] += 1
            elif d.detectionCorrect is False:
                incorrectPerRoom[d.room] += 1

        print("\nCorrect/Incorrect detections per room:")
        allRooms = sorted(set(correctPerRoom) | set(incorrectPerRoom))
        for room in allRooms:
            c = correctPerRoom.get(room, 0)
            w = incorrectPerRoom.get(room, 0)
            ratio = f"{c}/{c+w}" if (c+w) > 0 else "n/a"
            print(f"  {room:20s} : {c} correct, {w} incorrect   (accuracy: {ratio})")

        # Detections per image
        countPerImage = Counter(d.imageId for d in visualDetections)
        print("\nTop 5 images by detection count:")
        for img, count in countPerImage.most_common(5):
            print(f"  {img:20s} : {count}")

        # Detection distances
        distances = np.array([d.distance for d in visualDetections])
        print("\nDetection distances [Distance]:")
        print(f"  min: {np.min(distances):.3f}")
        print(f"  max: {np.max(distances):.3f}")
        print(f"  mean: {np.mean(distances):.3f}")
        print(f"  std: {np.std(distances):.3f}")

        # Detections with empty or unknown DetectedObjectId
        unknownObjectId = [d for d in visualDetections if not d.detectedObjectId or d.detectedObjectId.strip() == ""]
        if unknownObjectId:
            print(f"\nNumber of detections with empty DetectedObjectId: {len(unknownObjectId)}")
            print("Examples:")
            for d in unknownObjectId[:3]:
                print(f"  DetectionId: {d.detectionId}, ImageId: {d.imageId}")

        # Show a few correct/incorrect examples
        correctExamples = [d for d in visualDetections if d.detectionCorrect is True][:3]
        incorrectExamples = [d for d in visualDetections if d.detectionCorrect is False][:3]
        print("\nSample correct detections:")
        for d in correctExamples:
            print(f"  DetectionId: {d.detectionId}, Label: {d.label}, Room: {d.room}, ImageId: {d.imageId}")

        print("\nSample incorrect detections:")
        for d in incorrectExamples:
            print(f"  DetectionId: {d.detectionId}, Label: {d.label}, Room: {d.room}, ImageId: {d.imageId}")