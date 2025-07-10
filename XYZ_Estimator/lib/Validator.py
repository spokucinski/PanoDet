from typing import List
from lib.GtEntry import GtEntry
from lib.VisualDetection import VisualDetection
from lib.RadioDetection import RadioPrediction
from consts.ObjectClasses import CODE55_CLASSES

class Validator:
    @staticmethod
    def validateClassLabels(gtEntries: List[GtEntry], expectedLabels: set) -> None:
        """
        Checks whether all class labels used in the ground truth entries are recognized.

        Args:
            gtEntries (List[GtEntry]): List of ground truth entries to check.
            expectedLabels (set): Set of allowed/recognized class labels.

        Output:
            Prints to the console a report of any unrecognized class labels,
            or a summary if all are valid.
        """
        notFound = []
        for gtEntry in gtEntries:
            if gtEntry.classLabel not in expectedLabels:
                notFound.append((gtEntry.objectId, gtEntry.classLabel))
        
        if notFound:
            print("Unrecognized class labels found in Ground Truth:")
            for objectId, label in notFound:
                print(f"  ObjectId: {objectId}  |  class label: '{label}'")
        else:
            print("All class labels in Ground Truth are valid.")

    @staticmethod
    def validateDetectionLabels(
        visualDetections: List[VisualDetection], 
        expectedClasses: set
        ) -> None:
        """
        Validates that all detection labels are present in the expected class set.

        Args:
            visualDetections (List[Detection]): List of visual detection entries.
            expectedClasses (set): Set of valid class labels.
        
        Output:
            Prints a report of unrecognized class labels or a summary if all are valid.
        """
        unrecognizedClasses = []
        
        for detection in visualDetections:
            if detection.label not in expectedClasses:
                unrecognizedClasses.append(
                    (detection.relativeDetectionNumber, detection.label))
                
        if unrecognizedClasses:
            print("Unrecognized class labels in detections:")
            for rel_num, label in unrecognizedClasses:
                print(f"  RelDetNumber: {rel_num}  |  Label: '{label}'")
        else:
            print("All class labels in detections are valid.")

    @staticmethod
    def validateDetectedObjectIds(
        visualDetections: List[VisualDetection],
        gtEntries: List[GtEntry]
    ) -> None:
        """
        Validates that all detected object IDs in visual detections match objects from ground truth.

        Args:
            detections (List[VisualDetection]): List of visual detection entries.
            gtEntries (List[GtEntry]): List of ground truth entries.

        Output:
            Prints a report of unmatched object IDs or a summary if all are valid.
        """
        allGtObjectIds = {entry.objectId for entry in gtEntries}
        
        unmatched = [
            (detection.detectionId, detection.detectedObjectId)
            for detection in visualDetections
            if detection.detectedObjectId not in allGtObjectIds
        ]

        if unmatched:
            print(f"Some detections (Count: {len(unmatched)}) refer to unrecognized ObjectIds!")
            print("Unmatched ObjectIds:")
            for detectionId, objectId in unmatched:
                print(f"  DetectionId: {detectionId}  |  DetectedObjectId: '{objectId}'")
        else:
            print("All ObjectIds from visual detections match with objects from ground truth")

    @staticmethod
    def validateRadioObjectIds(
        radioDetections: List[RadioPrediction],
        gtEntries: List[GtEntry]
    ) -> None:
        gtObjectIds = {entry.objectId for entry in gtEntries}
        unmatched = [
            (radio.radioId, radio.objectId)
            for radio in radioDetections
            if radio.objectId not in gtObjectIds
        ]

        matched = len(radioDetections) - len(unmatched)
        print(f"Number of radio detections with matched ObjectId: {matched}")
        print(f"Number of radio detections with unmatched ObjectId: {len(unmatched)}")

        if unmatched:
            print("Unmatched radio detection examples:")
            for radio_id, object_id in unmatched:
                print(f"  RadioId: {radio_id}  |  ObjectId: '{object_id}'")
