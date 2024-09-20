import json
import paho.mqtt.client as mqtt
from azure.eventgrid import EventGridPublisherClient
from azure.core.credentials import AzureKeyCredential
from datetime import datetime

# Configuration
MQTT_BROKER = 'your_mosquitto_broker_address'
MQTT_PORT = 1883
MQTT_TOPIC = 'your/mqtt/topic'
EVENT_GRID_ENDPOINT = 'https://your_event_grid_endpoint'
EVENT_GRID_KEY = 'your_event_grid_key'
EVENT_GRID_TOPIC = 'your_event_grid_topic'

# Initialize Event Grid client
credential = AzureKeyCredential(EVENT_GRID_KEY)
event_grid_client = EventGridPublisherClient(EVENT_GRID_ENDPOINT, credential)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to Mosquitto MQTT Broker")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        message = msg.payload.decode()
        analysis_result = analyze_message(message)
        publish_to_event_grid(analysis_result)
    except Exception as e:
        print(f"Error processing message: {e}")

def analyze_message(message):
    distances = message.split(';')
    readings = {dist.split(':')[0]: float(dist.split(':')[1]) for dist in distances}
    
    non_zero_readings = sum(1 for value in readings.values() if value > 0)
    total_readings = len(readings)
    
    if total_readings != 8:
        raise ValueError(f"Incorrect number of readings: {total_readings}")
    
    correctness = non_zero_readings / total_readings
    classification = "Fully Correct" if non_zero_readings == 8 else f"{non_zero_readings}/8 Correct"
    
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "readings": readings,
        "non_zero_readings": non_zero_readings,
        "classification": classification,
        "correctness": correctness
    }
    
    return result

def publish_to_event_grid(result):
    event = {
        "id": "ID-" + datetime.utcnow().isoformat(),
        "eventType": "IndoorLocalization.Analysis",
        "subject": "IndoorLocalization",
        "eventTime": result['timestamp'],
        "data": result,
        "dataVersion": "1.0"
    }
    event_grid_client.send([event])
    print("Event published to Azure Event Grid")

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()
