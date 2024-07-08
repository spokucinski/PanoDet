import subprocess
import re as regExp
import concurrent.futures
import time

from secrets import TOPIC_ENDPOINT, TOPIC_KEY
from secrets import CHIP_PATH, CHIP_STORAGE
from secrets import TRACKER_NUM, TARGET_MEASUREMENTS, MAX_ERRORS
from typing import List, Optional, Tuple
from azure.eventgrid import EventGridPublisherClient, EventGridEvent
from azure.core.credentials import AzureKeyCredential
from distanceMeasurement import DistanceMeasurement

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

def parseDiagnosticLog(commandOutput: str) -> Optional[List[DistanceMeasurement]]:
    match = regExp.search(r"logContent:\s*([0-9a-fA-F]+)", commandOutput)
    if not match:
        return None
    logTextInHex = match.group(1)
    logTextInAscii = convertHexToAscii(logTextInHex)
    measurements = extractMeasurements(logTextInAscii)
    return measurements

def convertHexToAscii(stringInHex: str) -> str:
    bytesObject = bytes.fromhex(stringInHex)
    stringInAscii = bytesObject.decode("ASCII")
    return stringInAscii

def extractMeasurements(logText: str) -> List[DistanceMeasurement]:
    pattern = regExp.compile(r'A(\d):(\d{2}\.\d{2});') # Syntax: A1:XX.XX;A2:XX.XX;A3...
    foundMatches = pattern.findall(logText)
    measurements = [DistanceMeasurement(anchorId, float(distance)) for anchorId, distance in foundMatches]
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

def getNodeMeasurements(nodeId: int, chipToolPath: str, chipBaseStoragePath: str) -> Tuple[List[DistanceMeasurement], List[str]]:
    measurements = []
    errors = []
    errorCount = 0

    while len(measurements) < TARGET_MEASUREMENTS and errorCount < MAX_ERRORS:
        output, error = retrieveDiagnosticLogs(nodeId, chipToolPath, chipBaseStoragePath)
        if error or parseDiagnosticLog(output) is None:
            if error:
                note = getErrorType(error)
                errors.append(f"{error} (Note: {note})")
            else:
                errors.append("Unknown error")
            errorCount += 1
            if errorCount % 1 == 0:
                print(f"Error count {errorCount} for node {nodeId}: {error} (Note: {note})")
            continue

        node_measurements = parseDiagnosticLog(output)
        if node_measurements:
            measurements.extend(node_measurements[:TARGET_MEASUREMENTS - len(measurements)])

    if errorCount >= MAX_ERRORS:
        print(f"Node {nodeId} reached the maximum error limit of {MAX_ERRORS} errors.")

    return measurements, errors

def getErrorType(commandErrorOutput: str) -> str:
    if "CHIP Error 0x00000032: Timeout" in commandErrorOutput:
        return "Timeout"
    
    return "Error"

def main():
    allMeasurements = {}
    allErrors = {}

    print(f"Starting measurement collection from: {TRACKER_NUM} trackers!")
    with concurrent.futures.ThreadPoolExecutor(max_workers=TRACKER_NUM) as threadExecutor:
        futures = {threadExecutor.submit(getNodeMeasurements, nodeId, CHIP_PATH, CHIP_STORAGE): nodeId for nodeId in range(1, TRACKER_NUM + 1)}

        for future in concurrent.futures.as_completed(futures):
            nodeId = futures[future]
            try:
                measurements, errors = future.result()
                allMeasurements[nodeId] = measurements
                allErrors[nodeId] = errors
            except Exception as e:
                print(f"Node {nodeId} generated an exception: {e}")

    for nodeId, measurements in allMeasurements.items():
        print(f"Node {nodeId} collected {len(measurements)} measurements.")

    for nodeId, errors in allErrors.items():
        print(f"Node {nodeId} had {len(errors)} errors.")

    for nodeId, measurements in allMeasurements.items():
        reportMeasurementsToEventGrid(measurements, TOPIC_ENDPOINT, TOPIC_KEY)
        print(f"Measurements for node: {nodeId} sent to Azure Event Grid.")

if __name__ == "__main__":
    main()