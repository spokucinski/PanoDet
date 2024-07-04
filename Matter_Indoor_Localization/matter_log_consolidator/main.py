import subprocess
import re
from typing import List
from azure.eventgrid import EventGridPublisherClient, EventGridEvent
from azure.core.credentials import AzureKeyCredential

class DistanceMeasurement:
    def __init__(self, anchor_id: str, distance: float):
        self.anchor_id = anchor_id
        self.distance = distance

    def to_dict(self):
        return {"anchor_id": self.anchor_id, "distance": self.distance}

def run_chip_tool_command():
    # Run the chip-tool command in single command mode
    command = [ 
                "/home/RaspberryC/esp-matter/connectedhomeip/out/host/chip-tool",
                "diagnosticlogs",
                "retrieve-logs-request",
                "0",
                "0",
                "1",
                "1",
                "--storage-directory",
                "/home/RaspberryC/chipStorage"
            ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running chip-tool command: {result.stderr}")
        return None
    return result.stdout

def parse_log_content(output) -> List[DistanceMeasurement]:
    # Use regex to extract the logContent from the output
    match = re.search(r"logContent:\s*([0-9a-fA-F]+)", output)
    if not match:
        print("No logContent found in the output.")
        return None
    log_content_hex = match.group(1)

    # Convert hex to ASCII
    log_content_ascii = hex_to_ascii(log_content_hex)

    # Extract distance measurements from the log content
    measurements = []
    pattern = re.compile(r'A(\d):(\d{2}\.\d{2});')
    matches = pattern.findall(log_content_ascii)
    for match in matches:
        anchor_id, distance = match
        measurements.append(DistanceMeasurement(anchor_id, float(distance)))
    
    return measurements

def hex_to_ascii(hex_string):
    # Convert hex to ASCII
    bytes_object = bytes.fromhex(hex_string)
    ascii_string = bytes_object.decode("ASCII")
    return ascii_string

def send_to_event_grid(measurements: List[DistanceMeasurement], topic_endpoint: str, access_key: str):
    # Convert the list of DistanceMeasurement objects to a list of dictionaries
    data = [measurement.to_dict() for measurement in measurements]

    # Create the event grid event
    event = EventGridEvent(
        subject="measurements",
        event_type="measurements.received",
        data=data,
        data_version="1.0"
    )

    # Create the client
    credential = AzureKeyCredential(access_key)
    client = EventGridPublisherClient(topic_endpoint, credential)
    
    # Send the event to the Event Grid
    client.send([event])
    print("Log content successfully sent to Azure Event Grid.")


def main():
    topic_endpoint = ""
    access_key = ""
    
    # Run the chip-tool command
    output = run_chip_tool_command()
    if output is None:
        return

    # Check for errors in the output
    if "Error" in output:
        print(f"Error found in chip-tool output: {output}")
        return

    # Parse the log content
    measurements = parse_log_content(output)
    if measurements is None:
        return

    # Send the log content to Azure Event Grid
    send_to_event_grid(measurements, topic_endpoint, access_key)

if __name__ == "__main__":
    main()