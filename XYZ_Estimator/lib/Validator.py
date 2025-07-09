from typing import List
from lib.GtEntry import GtEntry
from lib.Detection import Detection
from lib.RadioDetection import RadioDetection
from consts.ObjectClasses import CODE55_CLASSES

class Validator:
    @staticmethod
    def validateGtClasses(gtEntries: List[GtEntry]) -> None:
        notFoundClasses = []
        
        for gtEntry in gtEntries:
            if gtEntry.code55Class not in CODE55_CLASSES:
                notFoundClasses.append((gtEntry.objectId, gtEntry.code55Class))
        
        if notFoundClasses:
            print("Nierozpoznane klasy CODE55 w Ground Truth:")
            for obj_id, code in notFoundClasses:
                print(f"  ObjectId: {obj_id}  |  CODE55: '{code}'")
        else:
            print("Wszystkie klasy CODE55 w GT są prawidłowe.")

    @staticmethod
    def validateDetectionLabels(detections: List[Detection]) -> None:
        unrecognizedClasses = []
        
        for detection in detections:
            if detection.label not in CODE55_CLASSES:
                unrecognizedClasses.append((detection.relativeDetectionNumber, detection.label))
                
        if unrecognizedClasses:
            print("Unrecognized class labels in detections:")
            for rel_num, label in unrecognizedClasses:
                print(f"  RelDetNumber: {rel_num}  |  Label: '{label}'")
        else:
            print("All class labels in detections are valid.")

    @staticmethod
    def validateDetectedObjectIds(
        detections: List[Detection],
        gtEntries: List[GtEntry]
    ) -> None:
        allGtObjectIds = {entry.objectId for entry in gtEntries}
        
        unmatched = [
            (detection.detectionId, detection.detectedObjectId)
            for detection in detections
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
        radioDetections: List[RadioDetection],
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
