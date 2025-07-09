from typing import Tuple, List
import pandas as pd
from lib.GtEntry import GtEntry
from lib.Detection import Detection
from lib.RadioDetection import RadioDetection

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
    def readGroundTruth(path: str) -> List[GtEntry]:
        dataFrame = pd.read_csv(path, sep=';')
        
        gtEntries = [
            GtEntry(
                groundTruthId=int(row['GroundTruthId']),
                room=row['Room'],
                collection=row['Collection'],
                objectId=row['ObjectId'],
                code55Class=row['CODE55'],
                xgt=float(row['Xgt']),
                ygt=float(row['Ygt']),
                zgt=float(row['Zgt'])
            )
            for _, row in dataFrame.iterrows()
        ]

        return gtEntries

    @staticmethod
    def readRadioPredictions(path: str) -> List[RadioDetection]:
        df = pd.read_csv(path, sep=';')
        entries = [
            RadioDetection(
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
            for _, row in df.iterrows()
        ]
        return entries

    @staticmethod
    def readVisualDetections(path: str) -> Tuple[List[Detection], int, int, int, List[int]]:
        dataFrame = pd.read_csv(path, sep=';')
        detections = []

        for _, row in dataFrame.iterrows():
            bool_val = DataLoader.parseStringToBool(row['DetectionCorrect'])
            
            detection = Detection(
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
