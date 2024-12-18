import subprocess
import re as regExp
import concurrent.futures
import time
import os

from datetime import datetime
from secrets import TOPIC_ENDPOINT, TOPIC_KEY
from secrets import CHIP_PATH, CHIP_STORAGE, OUT_BASE_PATH
from secrets import TRACKER_NUM, TARGET_MEASUREMENT_SETS, MAX_ERRORS
from typing import List, Optional, Tuple
from azure.eventgrid import EventGridPublisherClient, EventGridEvent
from azure.core.credentials import AzureKeyCredential
from distanceMeasurement import DistanceMeasurement
from error import Error

def retrieveDiagnosticLogs(nodeId: int, chipToolPath: str, chipStoragePath: str) -> Tuple[Optional[str], Optional[str]]:
    command = [
        chipToolPath,
        "diagnosticlogs",
        "retrieve-logs-request",
        "0",
        "0",
        f"{nodeId}",
        "1",
        "--storage-directory",
        chipStoragePath + str(nodeId)
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # return empty result and filled error
        return None, f"Error running chip-tool command for node {nodeId}: {result.stdout}"
    else:
        # return filled result and empty error
        return result.stdout, None

def parseDiagnosticLog(commandOutput: str, measurementSetExecutionTime: float) -> Optional[List[DistanceMeasurement]]:
    match = regExp.search(r"logContent:\s*([0-9a-fA-F]+)", commandOutput)
    if not match:
        return None
    logTextInHex = match.group(1)
    logTextInAscii = convertHexToAscii(logTextInHex)
    measurements = extractMeasurements(logTextInAscii, measurementSetExecutionTime)
    return measurements

def convertHexToAscii(stringInHex: str) -> str:
    bytesObject = bytes.fromhex(stringInHex)
    stringInAscii = bytesObject.decode("ASCII")
    return stringInAscii

def extractMeasurements(logText: str, measurementSetExecutionTime: float) -> List[DistanceMeasurement]:
    pattern = regExp.compile(r'A(\d):(\d{2}\.\d{2});?') # Syntax: A1:XX.XX;A2:XX.XX;A3...
    foundMatches = pattern.findall(logText)
    measurements = [DistanceMeasurement(anchorId, float(distance), measurementSetExecutionTime) for anchorId, distance in foundMatches]
    return measurements

def reportMeasurementsToEventGrid(measurements: List[DistanceMeasurement], eventGridTopicEndpoint: str, eventGridAccessKey: str):
    data = [measurement.to_dict() for measurement in measurements]
    event = EventGridEvent(
        subject="measurements",
        event_type="measurements.received",
        data=data,
        data_version="1.0"
    )
    credential = AzureKeyCredential(eventGridAccessKey)
    client = EventGridPublisherClient(eventGridTopicEndpoint, credential)
    client.send([event])

def getNodeMeasurements(trackerId: int, chipToolPath: str, chipBaseStoragePath: str) -> Tuple[List[DistanceMeasurement], List[str], float]:
    experimentStartTime = time.time()
    measurements = []
    errors = []
    errorCount = 0
    measurementSetCount = 0

    while measurementSetCount < TARGET_MEASUREMENT_SETS and errorCount < MAX_ERRORS:
        print(f"Tracker {trackerId} collects measurements, Now: {len(measurements)} measurements and {errorCount} errors.")
        
        measurementSetStartTime = time.time()
        output, error = retrieveDiagnosticLogs(trackerId, chipToolPath, chipBaseStoragePath)
        measurementSetEndTime = time.time()
        measurementSetExecutionTime = measurementSetEndTime - measurementSetStartTime

        # Error occured
        if error:
            errorType = getErrorType(error)
            errors.append(Error(errorType, measurementSetExecutionTime))
            errorCount += 1
            continue
        
        parsedMeasurements = parseDiagnosticLog(output, measurementSetExecutionTime)
        if parsedMeasurements is None:
            errorType = getErrorType(output)
            errors.append(Error(errorType, measurementSetExecutionTime))
            errorCount += 1
            continue

        measurements.extend(parsedMeasurements)
        measurementSetCount += 1

    if errorCount >= MAX_ERRORS:
        print(f"Tracker {trackerId} reached the maximum error limit of {MAX_ERRORS} errors.")

    experimentEndTime = time.time()
    experimentExecution = experimentEndTime - experimentStartTime

    print(f"Tracker {trackerId} finished experiment, done {measurementSetCount} sets, collected: {len(measurements)} measurements and {len(errors)} errors.")
    print(f"Tracker {trackerId} experiment took: {experimentExecution}")

    return measurements, errors, experimentExecution

def getErrorType(commandErrorOutput: str) -> str:
    if "CHIP Error 0x00000032: Timeout" in commandErrorOutput:
        return "Timeout"
    
    noLogsPattern = regExp.compile(r"RetrieveLogsResponse.*status:\s*2", regExp.DOTALL)
    if noLogsPattern.search(commandErrorOutput):
        return "NoLogs"
    
    return "Unknown Error"

def main():
    allMeasurements = {}
    allErrors = {}
    allExecutionTimes = {}

    print(f"Starting measurement collection from: {TRACKER_NUM} trackers!")
    # measurements, errors, experimentExecutionTime = getNodeMeasurements(2, CHIP_PATH, CHIP_STORAGE)
    with concurrent.futures.ThreadPoolExecutor(max_workers=TRACKER_NUM) as threadExecutor:
        futures = {threadExecutor.submit(getNodeMeasurements, trackerId, CHIP_PATH, CHIP_STORAGE): trackerId for trackerId in range(1, TRACKER_NUM + 1)}

        for future in concurrent.futures.as_completed(futures):
            trackerId = futures[future]
            measurements, errors, experimentExecutionTime = future.result()
            allMeasurements[trackerId] = measurements
            allErrors[trackerId] = errors
            allExecutionTimes[trackerId] = experimentExecutionTime 

    # Create output directories
    outputBaseDir = os.path.join(OUT_BASE_PATH, "out", "matter")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    outputDir = os.path.join(outputBaseDir, timestamp)
    os.makedirs(outputDir, exist_ok=True)

    for trackerId, measurements in allMeasurements.items():
        filename = os.path.join(outputDir, f"tracker_{trackerId}.txt")
        with open(filename, "w") as file:
            summary = f"Tracker {trackerId} collected {len(measurements)} measurements."
            print(summary)
            print(summary, file = file)
            for measurement in measurements:
                measurementSummary = f"Anchor ID: {measurement.anchorId}, Distance: {measurement.distance:.2f}, Exec Time: {measurement.executionTime:.2f}s"
                print(measurementSummary)
                print(measurementSummary, file = file)

    for trackerId, errors in allErrors.items():
        filename = os.path.join(outputDir, f"tracker_{trackerId}.txt")
        with open(filename, "a") as file:
            summary = f"Tracker {trackerId} had {len(errors)} errors."
            print(summary)
            print(summary, file = file)
            for error in errors:
                errorSummary = f"Error: {error.errorType}, Exec Time: {error.executionTime:.2f}s"
                print(errorSummary)
                print(errorSummary, file = file)

    for trackerId, executionTime in allExecutionTimes.items():
        filename = os.path.join(outputDir, f"tracker_{trackerId}.txt")
        with open(filename, "a") as file:
            executionTimeSummary = f"Whole experiment took: {executionTime:.2f}s"
            print(executionTimeSummary)
            print(executionTimeSummary, file = file)

    # for trackerId, measurements in allMeasurements.items():
    #     reportMeasurementsToEventGrid(measurements, TOPIC_ENDPOINT, TOPIC_KEY)
    #     print(f"Measurements for Tracker: {trackerId} sent to Azure Event Grid.")

if __name__ == "__main__":
    main()