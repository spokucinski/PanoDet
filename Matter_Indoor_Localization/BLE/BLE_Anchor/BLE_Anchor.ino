#include <WiFi.h>
#include <PubSubClient.h>
#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <Adafruit_NeoPixel.h>

////// ##### MANUAL SETUP ##### 
String _ssid = "";        // Here provide the local Wifi SSID
String _password = "";    // Here provide the local Wifi Password
String _broker = "";      // Here provide the local IP of message broker
const int _mqttPort = 0;  // Here provide the mqtt port the messages are broadcasted on

String _anchorId = "BLE_Anchor_1";
String _mqttStatusTopic = "BLE_Status";

const long _aliveInterval = 10000;  // miliseconds of interval between alive report

// Set the advertising interval (units of 0.625ms, i.e., 48 * 0.625ms = 30ms)
uint16_t _minAdvInterval = 0x0020;  // 20ms
uint16_t _maxAdvInterval = 0x0020;  // 20ms
////// ##### MANUAL SETUP ##### 

////// GLOBALS
WiFiClient _espClient;
PubSubClient _mqttClient(_espClient);
unsigned long _lastStatusTime = 0;
String _bleMacAddress;
Adafruit_NeoPixel _builtInLed = Adafruit_NeoPixel(1, 8, NEO_GRB + NEO_KHZ800); // ESP32C6's LED in on IO PIN 8

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

void initializeLed(){
  _builtInLed.begin();
  _builtInLed.setBrightness(10);
  _builtInLed.show();
}

void setRGBColor(int r, int g, int b) {
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

      if (_mqttClient.connect(_anchorId.c_str())) 
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

  Serial.println("Started BLE initialization!");
  // Init device
  BLEDevice::init(_anchorId);

  // Init server
  BLEDevice::createServer();

  // Create advertising
  BLEAdvertising* pAdvertising = BLEDevice::getAdvertising();
  
  // Set advertisement data (flags, etc.)
  BLEAdvertisementData oAdvertisementData = BLEAdvertisementData();
  oAdvertisementData.setFlags(0x04);  // Set BLE general discoverable mode
  pAdvertising->setAdvertisementData(oAdvertisementData);
  pAdvertising->setMinInterval(_minAdvInterval);
  pAdvertising->setMaxInterval(_maxAdvInterval);
  pAdvertising->addServiceUUID(BLEUUID((uint16_t)0x181A)); // Example service UUID - sensor
  pAdvertising->setScanResponse(true);

  pAdvertising->start();

  // Print the MAC address
  _bleMacAddress = BLEDevice::getAddress().toString();
  Serial.println(_anchorId + " MAC Address: " + _bleMacAddress);

  Serial.println("BLE initialization finished!");
}

void reportAlive()
{
  String message = _anchorId + " alive!";
  _mqttClient.publish(_mqttStatusTopic.c_str(), message.c_str());
}

void setup() 
{
  initializeLed(); 
  
  initializeSerial();

  Serial.println("Starting setup of " + _anchorId);

  initializeBLE(); // blue

  connectToWiFi(); // white

  connectToMQTT(); // red

  // Report MAC Address to server
  String message = _anchorId + " MAC Address: " + _bleMacAddress;
  _mqttClient.publish(_mqttStatusTopic.c_str(), message.c_str());
  
  Serial.println("Setup of " + _anchorId + " finished!");
}

void loop() 
{
  if (!_mqttClient.connected()) {
    connectToMQTT();
  }
  _mqttClient.loop();

  // Check if reporting period elapsed
  unsigned long currentMillis = millis();
  if (currentMillis - _lastStatusTime >= _aliveInterval) {
    setRGBColor(0, 255, 0);  // Green
    _lastStatusTime = currentMillis;
    reportAlive();
  }
}
