import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from typing import List
from lib.GtEntry import GtEntry
from consts.ObjectClasses import CODE55_CLASSES
from lib.VisualDetection import VisualDetection
import numpy as np

class Visualizer:
    @staticmethod
    def saveGtCountPerRoomHistogram(
        gtEntries: List[GtEntry], 
        outDir: str, 
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)
        countPerRoom = Counter(entry.room for entry in gtEntries)
        sortedRoomItems = sorted(countPerRoom.items(), key=lambda x: x[1], reverse=True)
        if not sortedRoomItems:
            return
        rooms, roomCounts = zip(*sortedRoomItems)
        if polish:
            title = "Liczba obiektów na pokój"
            xlabel = "Pokój"
            ylabel = "Liczba obiektów"
            fileName = "GtCountPerRoom_PL.png"
        else:
            title = "Object Count per Room"
            xlabel = "Room"
            ylabel = "Number of Objects"
            fileName = "GtCountPerRoom_ENG.png"
        plt.figure(figsize=(max(6, len(rooms) * 0.9), 5))
        plt.bar(rooms, roomCounts, color='skyblue', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, fileName))
        plt.close()

    @staticmethod
    def saveGtCountPerClassExistingHistogram(
        gtEntries: List[GtEntry], 
        outDir: str, 
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)
        countPerClass = Counter(entry.classLabel for entry in gtEntries)
        sortedClassItems = sorted(countPerClass.items(), key=lambda x: x[1], reverse=True)
        if not sortedClassItems:
            return
        classesExisting, classCountsExisting = zip(*sortedClassItems)
        if polish:
            title = "Liczba obiektów na klasę (występujące w GT)"
            xlabel = "Klasa obiektu"
            ylabel = "Liczba obiektów"
            fileName = "GtCountPerClassExisting_PL.png"
        else:
            title = "Object Count per Class (Existing Only)"
            xlabel = "Class Label"
            ylabel = "Number of Objects"
            fileName = "GtCountPerClassExisting_ENG.png"
        plt.figure(figsize=(max(8, len(classesExisting) * 0.38), 5))
        plt.bar(classesExisting, classCountsExisting, color='orange', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=70, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, fileName))
        plt.close()

    @staticmethod
    def saveGtCountPerClassAllHistogram(
        gtEntries: List[GtEntry],
        allClassLabels: List[str],
        outDir: str,
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)
        countPerClass = Counter(entry.classLabel for entry in gtEntries)
        countPerAllClasses = {c: countPerClass.get(c, 0) for c in allClassLabels}
        sortedAllClasses = sorted(countPerAllClasses.items(), key=lambda x: x[1], reverse=True)
        if not sortedAllClasses:
            return
        allClassesSorted, allCountsSorted = zip(*sortedAllClasses)
        if polish:
            title = "Liczba obiektów na klasę (wszystkie klasy)"
            xlabel = "Klasa obiektu"
            ylabel = "Liczba obiektów"
            fileName = "GtCountPerClassAll_PL.png"
        else:
            title = "Object Count per Class (All Classes)"
            xlabel = "Class Label"
            ylabel = "Number of Objects"
            fileName = "GtCountPerClassAll_ENG.png"
        plt.figure(figsize=(max(8, len(allClassesSorted) * 0.38), 5))
        plt.bar(allClassesSorted, allCountsSorted, color='green', edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=70, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(outDir, fileName))
        plt.close()

    @staticmethod
    def saveDetectionCorrectnessPie(
        visualDetections: List[VisualDetection], 
        outDir: str, 
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)
        total = len(visualDetections)
        correctlyRecognizedCount = sum(1 for d in visualDetections if d.detectionCorrect is True)
        wronglyRecognizedCount = sum(1 for d in visualDetections if d.detectionCorrect is False)
        invalidCount = total - (correctlyRecognizedCount + wronglyRecognizedCount)

        if polish:
            title = f"Poprawność detekcji z {total} próbek"
            labels = []
            colors = []
            if correctlyRecognizedCount > 0:
                labels.append("Poprawne")
                colors.append("green")
            if wronglyRecognizedCount > 0:
                labels.append("Niepoprawne")
                colors.append("red")
            if invalidCount > 0:
                labels.append("Nieprawidłowe")
                colors.append("gray")
        else:
            title = f"Overall detection correctness out of {total} samples"
            labels = []
            colors = []
            if correctlyRecognizedCount > 0:
                labels.append("Correct")
                colors.append("green")
            if wronglyRecognizedCount > 0:
                labels.append("Incorrect")
                colors.append("red")
            if invalidCount > 0:
                labels.append("Invalid")
                colors.append("gray")

        sizes = []
        value_list = []
        if correctlyRecognizedCount > 0:
            sizes.append(correctlyRecognizedCount)
            value_list.append(correctlyRecognizedCount)
        if wronglyRecognizedCount > 0:
            sizes.append(wronglyRecognizedCount)
            value_list.append(wronglyRecognizedCount)
        if invalidCount > 0:
            sizes.append(invalidCount)
            value_list.append(invalidCount)

        def my_autopct(values):
            def format_func(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                if total == 0:
                    return "0\n0.0%"
                return f"{val}\n{pct:.1f}%"
            return format_func

        plt.figure(figsize=(6, 6))
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct=my_autopct(value_list),
            startangle=90,
            textprops=dict(color="black", fontsize=13)
        )
        plt.axis('equal')
        plt.title(title, fontsize=16, pad=35)
        plt.tight_layout()
        lang_code = "PL" if polish else "ENG"
        plt.savefig(os.path.join(outDir, f"OverallDetectionCorrectness_{lang_code}.png"))
        plt.close()

    @staticmethod
    def saveDetectionsPerRoomHistogram(
        visualDetections: List[VisualDetection], 
        outDir: str, 
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)
        countPerRoom = Counter(d.room for d in visualDetections)
        sortedRooms = sorted(countPerRoom.items(), key=lambda x: x[1], reverse=True)
        rooms, roomCounts = zip(*sortedRooms)
        totalDetections = sum(roomCounts)
        percentages = [count / totalDetections * 100 for count in roomCounts]

        # Switch chart texts between English and Polish
        if polish:
            title = "Liczba detekcji wizualnych w pokojach"
            xlabel = "Pokój"
            ylabel = "Liczba detekcji wizualnych"
            totalLabel = f'Łączna liczba detekcji: {totalDetections}'
            lang_code = "PL"
        else:
            title = "Visual Detections per Room"
            xlabel = "Room"
            ylabel = "Number of Visual Detections"
            totalLabel = f'Detections in total: {totalDetections}'
            lang_code = "ENG"

        fig, ax = plt.subplots(figsize=(max(6, len(rooms) * 0.9), 6))
        bars = ax.bar(rooms, roomCounts, color='skyblue', edgecolor='black')
        ax.set_ylim(0, max(roomCounts) * 1.08)

        # Add percentage labels on top of each bar
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, color='blue', fontweight='bold')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=30, ha='right')

        # Total annotation
        ax.text(
            0.98, 0.95,
            totalLabel,
            ha='right', va='top',
            transform=ax.transAxes,
            fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        plt.tight_layout()
        out_file = f"VisualDetectionsPerRoom_{lang_code}.png"
        plt.savefig(os.path.join(outDir, out_file))
        plt.close()

    @staticmethod
    def saveCorrectIncorrectPerRoomHistogram(
        visualDetections: List[VisualDetection], 
        outDir: str, 
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)

        # Count correct/incorrect per room
        correct_counts = defaultdict(int)
        incorrect_counts = defaultdict(int)
        for d in visualDetections:
            if d.detectionCorrect is True:
                correct_counts[d.room] += 1
            elif d.detectionCorrect is False:
                incorrect_counts[d.room] += 1

        all_rooms = sorted(set(correct_counts) | set(incorrect_counts), key=lambda r: (-(correct_counts[r]+incorrect_counts[r]), r))
        correct = [correct_counts[r] for r in all_rooms]
        incorrect = [incorrect_counts[r] for r in all_rooms]

        total_correct = sum(correct)
        total_incorrect = sum(incorrect)

        # Texts for English/Polish
        if polish:
            title = "Poprawne i błędne detekcje w pokojach"
            xlabel = "Pokój"
            ylabel = "Liczba detekcji"
            legend_labels = ["Poprawne", "Błędne"]
            totalLabel = f'Łącznie poprawnych: {total_correct} | błędnych: {total_incorrect}'
            lang_code = "PL"
        else:
            title = "Correct and Incorrect Detections per Room"
            xlabel = "Room"
            ylabel = "Number of Detections"
            legend_labels = ["Correct", "Incorrect"]
            totalLabel = f'Total correct: {total_correct} | incorrect: {total_incorrect}'
            lang_code = "ENG"

        x = np.arange(len(all_rooms))
        width = 0.38

        fig, ax = plt.subplots(figsize=(max(6, len(all_rooms) * 1.2) + 2, 6))
        bars1 = ax.bar(x - width/2, correct, width, label=legend_labels[0], color='green', edgecolor='black')
        bars2 = ax.bar(x + width/2, incorrect, width, label=legend_labels[1], color='red', edgecolor='black')

        for bar, val in zip(bars1, correct):
            if val > 0:
                ax.annotate(f'{val}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10, color='green', fontweight='bold')
        for bar, val in zip(bars2, incorrect):
            if val > 0:
                ax.annotate(f'{val}',
                            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10, color='red', fontweight='bold')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(all_rooms, rotation=30, ha='right')
        ax.set_ylim(0, max(correct + incorrect) * 1.10)

        # Legend: place at right, vertical center
        ax.legend(
            loc="center right",
            bbox_to_anchor=(0.96, 0.5),  # (x, y) in axes fraction coordinates
            borderaxespad=0.5,
            fontsize=12,
            frameon=True,
            fancybox=True,
            facecolor='white'
        )

        # Total annotation (top right inside plot)
        ax.text(
            0.98, 0.95,
            totalLabel,
            ha='right', va='top',
            transform=ax.transAxes,
            fontsize=12, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        plt.tight_layout(rect=[0, 0, 0.90, 1])  # leave room for legend
        out_file = f"CorrectIncorrectPerRoom_{lang_code}.png"
        plt.savefig(os.path.join(outDir, out_file))
        plt.close()
    
    @staticmethod
    def saveDetectionsPerLabelHistogram(
        visualDetections: List[VisualDetection],
        outDir: str,
        topN = 15,
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)
        countPerLabel = Counter(d.label for d in visualDetections)
        sortedLabels = sorted(countPerLabel.items(), key=lambda x: x[1], reverse=True)

        # Select Top N
        if topN > 0 and topN < len(sortedLabels):
            shownLabels = sortedLabels[:topN]
        else:
            shownLabels = sortedLabels

        shown_label_list, shown_counts = zip(*shownLabels) if shownLabels else ([], [])
        totalShown = sum(shown_counts)

        # Language-specific text
        if polish:
            title = f"Liczba detekcji na klasę obiektu (Top {topN})" if topN > 0 else "Liczba detekcji na klasę obiektu (wszystkie)"
            xlabel = "Klasa obiektu"
            ylabel = "Liczba detekcji"
            totalLabel = f'Łączna liczba detekcji (Top {topN} klas): {totalShown}' if topN > 0 else f'Łączna liczba detekcji: {totalShown}'
            lang_code = "PL"
        else:
            title = f"Detections per Label/Class (Top {topN})" if topN > 0 else "Detections per Label/Class (ALL)"
            xlabel = "Label/Class"
            ylabel = "Number of Detections"
            totalLabel = f'Detections in Top {topN} classes: {totalShown}' if topN > 0 else f'All detections: {totalShown}'
            lang_code = "ENG"

        fig, ax = plt.subplots(figsize=(max(8, len(shown_label_list) * 0.38), 6))
        bars = ax.bar(shown_label_list, shown_counts, color='orange', edgecolor='black')
        ax.set_ylim(0, max(shown_counts)*1.08 if shown_counts else 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=70, ha='right', fontsize=9)

        # Add count labels on top of each bar
        for bar, count in zip(bars, shown_counts):
            height = bar.get_height()
            ax.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, color='blue', fontweight='bold')

        # Total annotation
        ax.text(
            0.98, 0.95,
            totalLabel,
            ha='right', va='top',
            transform=ax.transAxes,
            fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        plt.tight_layout()
        out_file = f"VisualDetectionsPerLabel_{lang_code}_Top{topN if topN else 'ALL'}.png"
        plt.savefig(os.path.join(outDir, out_file))
        plt.close()

    @staticmethod
    def saveDetectionCorrectnessPerClassBar(
        visualDetections: List[VisualDetection],
        outDir: str,
        topN = 15,
        polish: bool = False
    ):
        os.makedirs(outDir, exist_ok=True)

        correctPerLabel = defaultdict(int)
        incorrectPerLabel = defaultdict(int)
        for d in visualDetections:
            if d.detectionCorrect is True:
                correctPerLabel[d.label] += 1
            elif d.detectionCorrect is False:
                incorrectPerLabel[d.label] += 1

        allLabels = sorted(set(correctPerLabel) | set(incorrectPerLabel))
        allLabels = sorted(allLabels, key=lambda l: correctPerLabel.get(l, 0), reverse=True)

        if topN > 0 and topN < len(allLabels):
            allLabels = allLabels[:topN]

        corrects = [correctPerLabel.get(l, 0) for l in allLabels]
        incorrects = [incorrectPerLabel.get(l, 0) for l in allLabels]
        totalShown = sum(corrects) + sum(incorrects)

        x = np.arange(len(allLabels))
        width = 0.40

        if polish:
            title = f"Poprawne vs błędne detekcje na klasę (Top {topN})" if topN > 0 else "Poprawne vs błędne detekcje na klasę (wszystkie)"
            xlabel = "Klasa obiektu"
            ylabel = "Liczba detekcji"
            legend_labels = ["Poprawne", "Błędne"]
            totalLabel = f'Łączna liczba detekcji (Top {topN} klas): {totalShown}' if topN > 0 else f'Łączna liczba detekcji: {totalShown}'
            lang_code = "PL"
        else:
            title = f"Correct vs Incorrect Detections per Class (Top {topN})" if topN > 0 else "Correct vs Incorrect Detections per Class (ALL)"
            xlabel = "Label/Class"
            ylabel = "Number of Detections"
            legend_labels = ["Correct", "Incorrect"]
            totalLabel = f'Detections in Top {topN} classes: {totalShown}' if topN > 0 else f'All detections: {totalShown}'
            lang_code = "ENG"

        fig, ax = plt.subplots(figsize=(max(10, len(allLabels) * 0.45), 7))
        bars1 = ax.bar(x - width/2, corrects, width, label=legend_labels[0], color='green')
        bars2 = ax.bar(x + width/2, incorrects, width, label=legend_labels[1], color='red')
        ax.set_ylim(0, (max(corrects + incorrects) if corrects or incorrects else 1) * 1.08)
        ax.set_xticks(x)
        ax.set_xticklabels(allLabels, rotation=65, ha='right', fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        for rect in bars1:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color='green', fontweight='bold')
        for rect in bars2:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 4),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=9, color='red', fontweight='bold')

        # Total annotation
        ax.text(
            0.98, 0.95,
            totalLabel,
            ha='right', va='top',
            transform=ax.transAxes,
            fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
        )

        # Legend INSIDE the chart, right side, vertically centered
        ax.legend(
            loc="center right",
            fontsize=11,
            frameon=True,
            facecolor='white',
            bbox_to_anchor=(0.98, 0.5)
        )

        plt.tight_layout()
        out_file = f"VisualDetectionCorrectnessPerClass_{lang_code}_Top{topN if topN else 'ALL'}.png"
        plt.savefig(os.path.join(outDir, out_file))
        plt.close()

    @staticmethod
    def saveDetectionDistancesHistogram(
        visualDetections: List[VisualDetection], 
        outDir: str, 
        polish: bool = False, 
        correctOnly = False
    ):
        os.makedirs(outDir, exist_ok=True)

        # Filter distances based on correctOnly switch
        if correctOnly:
            distances = np.array([d.distance for d in visualDetections if d.detectionCorrect is True])
        else:
            distances = np.array([d.distance for d in visualDetections])
        
        if len(distances) == 0:
            print("No detection distances found; skipping histogram.")
            return

        bin_width = 0.125  # 12.5 cm
        min_dist = 0.0
        max_dist = np.ceil(distances.max() / bin_width) * bin_width
        bins = np.arange(min_dist, max_dist + bin_width, bin_width)

        # Labels & language
        if polish:
            title = "Rozkład odległości detekcji"
            xlabel = "Odległość [m]"
            ylabel = "Liczba detekcji"
            lang_code = "PL"
        else:
            title = "Detection Distances Distribution"
            xlabel = "Distance [m]"
            ylabel = "Number of Detections"
            lang_code = "ENG"

        if correctOnly:
            title += " (Correct Only)" if not polish else " (Tylko poprawne)"
            file_tag = "_CorrectOnly"
        else:
            file_tag = ""

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(distances, bins=bins, color='royalblue', edgecolor='black')

        # Set X-ticks every 0.5 m
        xticks = np.arange(min_dist, max_dist + 0.001, 0.5)
        ax.set_xticks(xticks)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        out_file = f"VisualDetectionDistances{file_tag}_{lang_code}.png"
        plt.savefig(os.path.join(outDir, out_file))
        plt.close()