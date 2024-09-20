#include <WiFi.h>
#include <PubSubClient.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEScan.h>
#include <Adafruit_NeoPixel.h>
#include <vector>
#include <math.h>

////// ##### MANUAL SETUP ##### 
String _ssid = "";              // Here provide the local Wifi SSID
String _password = "";          // Here provide the local Wifi Password
String _broker = "";            // Here provide the local IP of message broker
const int _mqttPort = 0;        // Here provide the mqtt port the messages are broadcasted on

String _trackerId = "BLE_Tracker_1";
String _mqttTrackingTopic = "BLE_Tracking/" + _trackerId;
String _mqttStatusTopic = "BLE_Status";

// Period between scanning procedures in the main loop of application
const int _scanInterval = 1000;   // miliseconds
// How long does a single full scanning procedure take
const int _bleScanTime = 1;        // seconds
// Internal parameter of scanning procedure - how long does the internal scan take
const int _bleScanInterval = 110; // miliseconds
// Internal parameter of scanning procedure - for how long (during the internal scan) 
// is the device listening for advertisements
const int _bleScanWindow = 100; // miliseconds

const int _aliveInterval = 10000;  // miliseconds
const bool _activeScan = true;

const int _txPower = -51;               // Value from device calibration, reference RSSI at 1 meter distance
const double _pathLossExponent = 1.45;  // Value from device calibration

const char* _knownAnchors[] = {
  "40:40:40:40:40:40",  // BA1 BleAnchorN
  "40:40:40:40:40:40",  // BA2
  "40:40:40:40:40:40",  // BA3
  "40:40:40:40:40:40",  // BA4
  "40:40:40:40:40:40",  // BA5
  "40:40:40:40:40:40",  // BA6
  "40:40:40:40:40:40",  // BA7
  "40:40:40:40:40:40"   // BA8
};
////// ##### MANUAL SETUP ##### 

////// GLOBALS
BLEScan* _BleScan;
WiFiClient _espClient;
PubSubClient _mqttClient(_espClient);
const int _numAnchors = sizeof(_knownAnchors) / sizeof(_knownAnchors[0]);
unsigned long _lastScanTime = 0;
unsigned long _lastStatusTime = 0;
Adafruit_NeoPixel _builtInLed = Adafruit_NeoPixel(1, 8, NEO_GRB + NEO_KHZ800);  // ESP32C6's LED in on IO PIN 8

struct AnchorReading {
  String address;
  int rssi;
  String anchorId;
  double distance;
};

std::vector<AnchorReading> _anchorReadings;

bool isKnownAnchor(const String& address) 
{
  for (int i = 0; i < _numAnchors; i++) {
    if (address.equalsIgnoreCase(_knownAnchors[i])) {
      return true;
    }
  }
  return false;
}

void initializeAnchorReadings() 
{
  for (int i = 0; i < _numAnchors; ++i) {
    AnchorReading anchorReading = { _knownAnchors[i], 0, "A" + String(i + 1), 0.0 };
    _anchorReadings.push_back(anchorReading);
  }
}

void updateAnchorRSSI(const String& address, int rssi) 
{
  for (AnchorReading& anchor : _anchorReadings) {
    if (anchor.address.equalsIgnoreCase(address)) {
      anchor.rssi = rssi;
      anchor.distance = calculateDistance(rssi);
      break;
    }
  }
}

// Function to calculate distance from RSSI using the Log-distance Path Loss Model
double calculateDistance(int rssi) 
{
  if (rssi == 0) {
    return 0;
  }

  if (_txPower == rssi) {
    return 0;
  }

  // Formula: Distance = 10 ^ ((txPower - RSSI) / (10 * pathLossExponent))
  double ratio = (_txPower - rssi) / (10.0 * _pathLossExponent);
  double distance = pow(10.0, ratio);

  return distance;
}

void connectToWiFi() 
{
  setRGBColor(255, 255, 255);  // White
  bool connected = false;
  while (!connected) {
    Serial.println("Attempting to connect to WiFi network: " + _ssid);
    WiFi.begin(_ssid, _password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) // Try for 20 iterations (10 seconds)
    {  
      delay(500);
      Serial.println("Connecting...");
      attempts++;
    }

    if (WiFi.status() == WL_CONNECTED)
    {
      connected = true;
    } 
    else 
    {
      Serial.println("Could not connect to " + _ssid);
    }
  }

  Serial.println("WiFi connected");
  Serial.println("IP address: " + WiFi.localIP().toString());
}

void initializeLed() 
{
  _builtInLed.begin();
  _builtInLed.setBrightness(10);
  _builtInLed.show();
}

void setRGBColor(int r, int g, int b) 
{
  _builtInLed.setPixelColor(0, _builtInLed.Color(r, g, b));
  _builtInLed.show();
}

void connectToMQTT() 
{
  setRGBColor(255, 0, 0);  // Red
  bool connected = false;
  while (!connected) 
  {
    Serial.println("Attempting to connect to MQTT broker: " + _broker);
    _mqttClient.setServer(_broker.c_str(), _mqttPort);

    while (!_mqttClient.connected()) 
    {
      // Check WiFi connection before attempting to connect to MQTT
      if (WiFi.status() != WL_CONNECTED) 
      {
        Serial.println("WiFi not connected during MQTT connection, attempting to reconnect to Wifi first...");
        connectToWiFi();
      }

      if (_mqttClient.connect(_trackerId.c_str())) 
      {
        connected = true;
      } 
      else 
      {
        Serial.println("Failed to connect to broker, state: " + String(_mqttClient.state()));
        Serial.println("Retrying in 500 ms...");
        delay(500);
      }
    }
  }

  Serial.println("MQTT connected");
}

void initializeSerial() 
{
  setRGBColor(255, 0, 255);  // Purple
  Serial.begin(115200);
  delay(5000);
}

void initializeBLE() 
{
  setRGBColor(0, 0, 255);  // Blue
  Serial.println("Initialization of BLE...");

  BLEDevice::init(_trackerId);
  _BleScan = BLEDevice::getScan();
  _BleScan->setActiveScan(_activeScan);
  _BleScan->setInterval(_bleScanInterval);
  _BleScan->setWindow(_bleScanWindow);

  Serial.println("BLE initialization finished!");
}

void scanForAnchors() 
{
  BLEScanResults* scanResults = _BleScan->start(_bleScanTime, false);

  // Reset readings of all known anchors
  for (auto& anchor : _anchorReadings) {
    anchor.rssi = 0;
    anchor.distance = 0.0;
  }

  if (scanResults) 
  {
    int foundDevices = scanResults->getCount();
    Serial.println("Scan returned: " + String(foundDevices) + " devices!");

    for (int i = 0; i < foundDevices; i++) {
      BLEAdvertisedDevice advertisedDevice = scanResults->getDevice(i);
      String address = advertisedDevice.getAddress().toString();

      if (isKnownAnchor(address)) {
        int rssi = advertisedDevice.getRSSI();
        Serial.println("Found known anchor: " + address + " RSSI: " + String(rssi));
        updateAnchorRSSI(address, rssi);
      }
    }
  }
}

String formatDistance(double distance){
  // Ensure distance is truncated to 99.99 if it exceeds this value
  if (distance > 99.99) {
    distance = 99.99;
  }

  // Ensure distance is not negative
  if (distance < 0.0) {
    distance = 0.0;
  }

  // Use default method for two decimal places
  String formattedReading = String(distance, 2);

  // If distance is lower than 10 meters - add 0 to the begining 
  if (formattedReading.length() < 5) {
    formattedReading = "0" + formattedReading;
  }

  return formattedReading;
}

void reportAnchorReadings() 
{
  String message = "";

  for (const AnchorReading& anchor : _anchorReadings) {
    message += anchor.anchorId + ":" + formatDistance(anchor.distance) + ";";
  }

  // Remove ending ';'
  if (message.length() > 0) {
    message.remove(message.length() - 1);
  }

  Serial.println(message);
  _mqttClient.publish(_mqttTrackingTopic.c_str(), message.c_str());
}

void reportAlive()
{
  String message = _trackerId + " alive!";
  _mqttClient.publish(_mqttStatusTopic.c_str(), message.c_str());
}

void setup() 
{
  initializeLed();

  initializeSerial();

  Serial.println("Starting setup of " + _trackerId);

  initializeBLE();

  connectToWiFi();

  connectToMQTT();

  initializeAnchorReadings();

  Serial.println("Setup of " + _trackerId + " finished!");
}

void loop() 
{
  if (!_mqttClient.connected()) 
  {
    connectToMQTT();
  }
  _mqttClient.loop();

  unsigned long currentMillis = millis();

  if (currentMillis - _lastScanTime >= _scanInterval) {
    setRGBColor(0, 255, 0);  // Green
    _lastScanTime = currentMillis;
    scanForAnchors();
    reportAnchorReadings();
  }

  if (currentMillis - _lastStatusTime >= _aliveInterval) {
    _lastStatusTime = currentMillis;
    reportAlive();
  }
}