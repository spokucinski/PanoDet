import argparse

from lib.Visualizer import Visualizer
from lib.Validator import Validator
from lib.DataLoader import DataLoader
from consts.ObjectClasses import CODE55_CLASSES

def main(gtPath, skipValidation=False):
    
    gtEntries = DataLoader.readGroundTruth(gtPath)

    if not skipValidation:
        Validator.validateGtClassLabels(gtEntries, CODE55_CLASSES)

    outDir = "data/out/gt"
    Visualizer.saveGtCountPerRoomHistogram(gtEntries=gtEntries,
                                           outDir=outDir,
                                           polish=False)
    
    Visualizer.saveGtCountPerRoomHistogram(gtEntries=gtEntries,
                                           outDir=outDir,
                                           polish=True)
    
    Visualizer.saveGtCountPerClassExistingHistogram(gtEntries=gtEntries,
                                                    outDir=outDir,
                                                    polish=False)
    
    Visualizer.saveGtCountPerClassExistingHistogram(gtEntries=gtEntries,
                                                    outDir=outDir,
                                                    polish=True)
    
    Visualizer.saveGtCountPerClassAllHistogram(gtEntries=gtEntries,
                                               allClassLabels=CODE55_CLASSES,
                                               outDir=outDir,
                                               polish=False)
    
    Visualizer.saveGtCountPerClassAllHistogram(gtEntries=gtEntries,
                                               allClassLabels=CODE55_CLASSES,
                                               outDir=outDir,
                                               polish=True)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Analyzer for ground truth data.")

    parser.add_argument("--gtPath", type=str, default="data/GroundTruth.csv", help="CSV file path for Ground Truth")
    parser.add_argument("--skipValidation", action="store_true", default=False, help="Skip input validation (default: False)")
    
    args = parser.parse_args()

    main(gtPath=args.gtPath, skipValidation=args.skipValidation)