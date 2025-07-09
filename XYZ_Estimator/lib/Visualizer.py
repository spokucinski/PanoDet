import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List
from lib.GtEntry import GtEntry
from consts.ObjectClasses import CODE55_CLASSES
from lib.Detection import Detection
import numpy as np

class Visualizer:
    @staticmethod
    def printGtStatistics(gtEntries: List[GtEntry]) -> None:
        print(f"########## Ground Truth Statistics ##########")
        
        total = len(gtEntries)
        print(f"Total number of samples in Ground Truth: {total}")

        countPerRoom = Counter(entry.room for entry in gtEntries)
        print("\nCount of objects per room:")
        for room, count in countPerRoom.items():
            print(f"  {room:20s} : {count}")

        countPerCODE55Class = Counter(entry.code55Class for entry in gtEntries)
        print("\nCount of objects per CODE55 class:")
        for code, count in countPerCODE55Class.items():
            print(f"  {code:20s} : {count}")

        print("\nMost common CODE55 class:", countPerCODE55Class.most_common(1))
        print("Most common room:", countPerRoom.most_common(1))

    @staticmethod
    def saveGtStatisticsHistograms(gtEntries: List[GtEntry], outDir: str = "data/out/statistics") -> None:
        os.makedirs(outDir, exist_ok=True)

        # Histogram: count per room, sorted descending
        countPerRoom = Counter(entry.room for entry in gtEntries)
        sortedRoomItems = sorted(countPerRoom.items(), key=lambda x: x[1], reverse=True)
        rooms, roomCounts = zip(*sortedRoomItems)
        plt.figure(figsize=(max(6, len(rooms) * 0.9), 5))
        plt.bar(rooms, roomCounts, color='skyblue', edgecolor='black')
        plt.title("Object Count per Room")
        plt.xlabel("Room")
        plt.ylabel("Number of Objects")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "GtCountPerRoom.png"))
        plt.close()

        # Histogram: CODE55 classes existing in GT, sorted descending
        countPerCODE55Class = Counter(entry.code55Class for entry in gtEntries)
        sortedClassItems = sorted(countPerCODE55Class.items(), key=lambda x: x[1], reverse=True)
        classesExisting, classCountsExisting = zip(*sortedClassItems)
        plt.figure(figsize=(max(8, len(classesExisting) * 0.38), 5))
        plt.bar(classesExisting, classCountsExisting, color='orange', edgecolor='black')
        plt.title("Object Count per CODE55 Class (Existing Only)")
        plt.xlabel("CODE55 Class")
        plt.ylabel("Number of Objects")
        plt.xticks(rotation=70, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "GtCountPerCode55ClassExisting.png"))
        plt.close()

        # Histogram: all CODE55 classes, including those missing from GT, sorted descending
        allCode55 = list(CODE55_CLASSES)
        countPerAllClasses = {c: countPerCODE55Class.get(c, 0) for c in allCode55}
        sortedAllClasses = sorted(countPerAllClasses.items(), key=lambda x: x[1], reverse=True)
        allClassesSorted, allCountsSorted = zip(*sortedAllClasses)
        plt.figure(figsize=(max(8, len(allClassesSorted) * 0.38), 5))
        plt.bar(allClassesSorted, allCountsSorted, color='green', edgecolor='black')
        plt.title("Object Count per CODE55 Class (All Classes)")
        plt.xlabel("CODE55 Class")
        plt.ylabel("Number of Objects")
        plt.xticks(rotation=70, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "GtCountPerCode55ClassAll.png"))
        plt.close()

        print(f"Histograms saved to {outDir}")

    @staticmethod
    def printVisualDetectionsStatistics(detections: List[Detection]) -> None:
        print("########## Visual Detections Statistics ##########")
        
        total = len(detections)
        print(f"Total number of visual detections: {total}")

        correctlyRecognizedCount = sum(1 for detection in detections if detection.detectionCorrect is True)
        wronglyRecognizedCount = sum(1 for detection in detections if detection.detectionCorrect is False)

        print(f"Number of correct detections (TRUE): {correctlyRecognizedCount}")
        print(f"Number of incorrect detections (FALSE): {wronglyRecognizedCount}")

        if total != correctlyRecognizedCount + wronglyRecognizedCount:
            print(f"Number of detections with invalid DetectionCorrect value: {total - (correctlyRecognizedCount + wronglyRecognizedCount)}")

        # Detections per room
        countPerRoom = Counter(d.room for d in detections)
        print("\nDetections per room:")
        for room, count in countPerRoom.items():
            print(f"  {room:20s} : {count}")

        # Detections per label/class
        countPerLabel = Counter(detection.label for detection in detections)
        print("\nDetections per label/class:")
        for label, count in countPerLabel.items():
            print(f"  {label:20s} : {count}")

        # Correct/incorrect ratio per class
        correctPerLabel = defaultdict(int)
        incorrectPerLabel = defaultdict(int)
        for d in detections:
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
        for d in detections:
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
        countPerImage = Counter(d.imageId for d in detections)
        print("\nTop 5 images by detection count:")
        for img, count in countPerImage.most_common(5):
            print(f"  {img:20s} : {count}")

        # Detection distances
        distances = np.array([d.distance for d in detections])
        print("\nDetection distances [Distance]:")
        print(f"  min: {np.min(distances):.3f}")
        print(f"  max: {np.max(distances):.3f}")
        print(f"  mean: {np.mean(distances):.3f}")
        print(f"  std: {np.std(distances):.3f}")

        # Detections with empty or unknown DetectedObjectId
        unknownObjectId = [d for d in detections if not d.detectedObjectId or d.detectedObjectId.strip() == ""]
        if unknownObjectId:
            print(f"\nNumber of detections with empty DetectedObjectId: {len(unknownObjectId)}")
            print("Examples:")
            for d in unknownObjectId[:3]:
                print(f"  DetectionId: {d.detectionId}, ImageId: {d.imageId}")

        # Show a few correct/incorrect examples
        correctExamples = [d for d in detections if d.detectionCorrect is True][:3]
        incorrectExamples = [d for d in detections if d.detectionCorrect is False][:3]
        print("\nSample correct detections:")
        for d in correctExamples:
            print(f"  DetectionId: {d.detectionId}, Label: {d.label}, Room: {d.room}, ImageId: {d.imageId}")

        print("\nSample incorrect detections:")
        for d in incorrectExamples:
            print(f"  DetectionId: {d.detectionId}, Label: {d.label}, Room: {d.room}, ImageId: {d.imageId}")
    
    @staticmethod
    def saveVisualDetectionsHistograms(
        detections: List[Detection],
        outDir: str = "data/out/statistics",
        topNImages: int = 20
    ) -> None:
        os.makedirs(outDir, exist_ok=True)

        # Pie chart: correct vs incorrect vs invalid detections
        total = len(detections)
        correctlyRecognizedCount = sum(1 for d in detections if d.detectionCorrect is True)
        wronglyRecognizedCount = sum(1 for d in detections if d.detectionCorrect is False)
        invalidCount = total - (correctlyRecognizedCount + wronglyRecognizedCount)

        labels = []
        sizes = []
        colors = []

        if correctlyRecognizedCount > 0:
            labels.append("Correct")
            sizes.append(correctlyRecognizedCount)
            colors.append("green")
        if wronglyRecognizedCount > 0:
            labels.append("Incorrect")
            sizes.append(wronglyRecognizedCount)
            colors.append("red")
        if invalidCount > 0:
            labels.append("Invalid")
            sizes.append(invalidCount)
            colors.append("gray")

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Overall Detection Correctness")
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "OverallDetectionCorrectness.png"))
        plt.close()

        # Histogram: detections per room (sorted)
        countPerRoom = Counter(detection.room for detection in detections)
        sortedRooms = sorted(countPerRoom.items(), key=lambda x: x[1], reverse=True)
        rooms, roomCounts = zip(*sortedRooms)
        plt.figure(figsize=(max(6, len(rooms) * 0.9), 5))
        plt.bar(rooms, roomCounts, color='skyblue', edgecolor='black')
        plt.title("Visual Detections per Room")
        plt.xlabel("Room")
        plt.ylabel("Number of Visual Detections")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "VisualDetectionsPerRoom.png"))
        plt.close()

        # Histogram: detections per label/class (sorted)
        countPerLabel = Counter(d.label for d in detections)
        sortedLabels = sorted(countPerLabel.items(), key=lambda x: x[1], reverse=True)
        labels, labelCounts = zip(*sortedLabels)
        plt.figure(figsize=(max(8, len(labels) * 0.38), 5))
        plt.bar(labels, labelCounts, color='orange', edgecolor='black')
        plt.title("Detections per Label/Class")
        plt.xlabel("Label/Class")
        plt.ylabel("Number of Detections")
        plt.xticks(rotation=70, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "VisualDetectionsPerLabel.png"))
        plt.close()

        # ---- CLASS-WISE CORRECT/INCORRECT GROUPED BAR CHART ----
        correctPerLabel = defaultdict(int)
        incorrectPerLabel = defaultdict(int)
        for d in detections:
            if d.detectionCorrect is True:
                correctPerLabel[d.label] += 1
            elif d.detectionCorrect is False:
                incorrectPerLabel[d.label] += 1

        allLabels = sorted(set(correctPerLabel) | set(incorrectPerLabel))
        corrects = [correctPerLabel.get(l, 0) for l in allLabels]
        incorrects = [incorrectPerLabel.get(l, 0) for l in allLabels]

        x = np.arange(len(allLabels))
        width = 0.40

        plt.figure(figsize=(max(10, len(allLabels) * 0.45), 6))
        plt.bar(x - width/2, corrects, width, label='Correct', color='green')
        plt.bar(x + width/2, incorrects, width, label='Incorrect', color='red')
        plt.xticks(x, allLabels, rotation=65, ha='right', fontsize=8)
        plt.xlabel('Label/Class')
        plt.ylabel('Number of Detections')
        plt.title('Correct vs Incorrect Detections per Class')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "VisualDetectionCorrectnessPerClass.png"))
        plt.close()

        # ---- CLASS-WISE ACCURACY BAR CHART ----
        accuracies = []
        for c, w in zip(corrects, incorrects):
            total = c + w
            acc = (c / total) if total > 0 else 0
            accuracies.append(acc * 100)  # percent

        plt.figure(figsize=(max(10, len(allLabels) * 0.45), 6))
        plt.bar(allLabels, accuracies, color='blue', edgecolor='black')
        plt.xticks(rotation=65, ha='right', fontsize=8)
        plt.xlabel('Label/Class')
        plt.ylabel('Accuracy [%]')
        plt.title('Class-wise Detection Accuracy')
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "VisualDetectionAccuracyPerClass.png"))
        plt.close()

        # Histogram: detections per image (top N)
        countPerImage = Counter(d.imageId for d in detections)
        mostCommonImages = countPerImage.most_common(topNImages)
        imageIds, imageCounts = zip(*mostCommonImages)
        plt.figure(figsize=(max(8, len(imageIds) * 0.5), 5))
        plt.bar(imageIds, imageCounts, color='green', edgecolor='black')
        plt.title(f"Detections per Image (Top {topNImages})")
        plt.xlabel("ImageId")
        plt.ylabel("Number of Detections")
        plt.xticks(rotation=60, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "VisualDetectionsPerImageTopN.png"))
        plt.close()

        # Histogram: detection distances
        distances = np.array([d.distance for d in detections])
        plt.figure(figsize=(8, 5))
        plt.hist(distances, bins=25, color='purple', edgecolor='black')
        plt.title("Detection Distances Distribution")
        plt.xlabel("Distance")
        plt.ylabel("Number of Detections")
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, "VisualDetectionDistances.png"))
        plt.close()

        print(f"Visual detection histograms saved to {outDir}")