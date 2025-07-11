from typing import Tuple, List
import pandas as pd
from lib.GtEntry import GtEntry
from lib.VisualDetection import VisualDetection
from lib.RadioDetection import RadioPrediction

class DataLoader:
    @staticmethod
    def parseStringToBool(val) -> bool:
        if isinstance(val, str):
            val_str = val.strip().upper()
            if val_str == "TRUE":
                return True
            if val_str == "FALSE":
                return False
        raise ValueError(f"Invalid value for boolean field: {val!r}")

    @staticmethod
    def readGroundTruth(groundTruthPath: str) -> List[GtEntry]:
        """
        Reads the ground truth data from a CSV file and converts each row into a GtEntry object.

        Args:
            groundTruthPath (str): Path to the GroundTruth.csv file.

        Returns:
            List[GtEntry]: A list of GtEntry objects parsed from the CSV file.

        Notes:
            - The CSV file must use a semicolon (';') as the separator.
            - All required columns (GroundTruthId, Room, Collection, ObjectId, CODE55, Xgt, Ygt, Zgt) must be present.
            - Each GtEntry holds the spatial ground truth information for one object.
        """
        dataFrame = pd.read_csv(groundTruthPath, sep=';')

        gtEntries = [
            GtEntry(
                groundTruthId=int(row['GroundTruthId']),
                room=row['Room'],
                collection=row['Collection'],
                objectId=row['ObjectId'],
                classLabel=row['CODE55'],
                xgt=float(row['Xgt']),
                ygt=float(row['Ygt']),
                zgt=float(row['Zgt'])
            )
            for _, row in dataFrame.iterrows()
        ]

        return gtEntries

    @staticmethod
    def readRadioPredictions(radioPredictionsPath: str) -> List[RadioPrediction]:
        dataFrame = pd.read_csv(radioPredictionsPath, sep=';')
        radioPredictions = [
            RadioPrediction(
                radioId=int(row['RadioId']),
                experiment=row['Experiment'],
                room=row['Room'],
                trackerId=row['TrackerId'],
                objectId=row['ObjectId'],
                xr=float(row['Xr']),
                yr=float(row['Yr']),
                zr=float(row['Zr']),
                xrgt=float(row['Xrgt']),
                yrgt=float(row['Yrgt']),
                zrgt=float(row['Zrgt'])
            )
            for _, row in dataFrame.iterrows()
        ]
        return radioPredictions

    @staticmethod
    def readVisualDetections(visionDetectionsPath: str) -> List[VisualDetection]:
        """
        Reads a CSV file containing vision detections and returns a list of Detection objects.
        
        Args:
            visionDetectionsPath (str): Path to the CSV file.
        
        Returns:
            List[Detection]: List of parsed Detection objects.
        """
        dataFrame = pd.read_csv(visionDetectionsPath, sep=';')
        detections = []

        for _, row in dataFrame.iterrows():
            bool_val = DataLoader.parseStringToBool(row['DetectionCorrect'])
            detection = VisualDetection(
                detectionId=int(row['DetectionId']),
                relativeDetectionNumber=int(row['RelDetNumber']),
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
                imageId=row['ImageId'],
                detectedObjectId=row['DetectedObjectId'],
                detectionCorrect=bool_val,
                detectedInRoom=row['DetectedInRoom']
            )
            detections.append(detection)

        return detections
