import subprocess

# Paths to your files
MAIN_SCRIPT = "calculateDQI.py"
GT_PATH = "data/GroundTruth.csv"
VISUAL_DETECTIONS_PATH = "data/DetectionsSummary.csv"
RADIO_PREDICTIONS_PATH = "data/RadioDetections.csv"

# Define all parameter arrays
radio_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
vision_weights = [1.0, 0.75, 0.5, 0.25, 0.0]
vision_sensitivities = [0.25, 0.5, 1.0, 2.0]
radio_sensitivities = [0.25, 0.5, 1.0, 2.0]
hybrid_modes = [True, False]

for i in range(len(radio_weights)):
    rw = radio_weights[i]
    vw = vision_weights[i]
    for vs in vision_sensitivities:
        for rs in radio_sensitivities:
            for hybrid in hybrid_modes:
                cmd = [
                    "python", MAIN_SCRIPT,
                    "--gtPath", GT_PATH,
                    "--visualDetectionsPath", VISUAL_DETECTIONS_PATH,
                    "--radioPredictionsPath", RADIO_PREDICTIONS_PATH,
                    "--radioWeight", str(rw),
                    "--visionWeight", str(vw),
                    "--radioSensitivity", str(rs),
                    "--visionSensitivity", str(vs),
                ]
                if hybrid:
                    cmd.append("--useHybridPositioning")
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)