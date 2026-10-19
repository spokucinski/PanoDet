# BLE subsystem

Arduino sketches for the BLE part of the MCIL framework, both targeting the Espressif ESP32-C6-DevKitC-1.

| Sketch | Role |
|---|---|
| `BLE_Anchor/BLE_Anchor.ino` | Anchor (reference point): continuous BLE advertising (20 ms interval), Wi-Fi connection and status reporting to the MQTT broker on `BLE_Status` |
| `BLE_Tracker/BLE_Tracker.ino` | Tracker (mobile device): active BLE scanning (110 ms interval, 100 ms window, ~1 s per scan cycle), RSSI-based distance estimation to every detected anchor, publishing of the estimates to `BLE_Tracking/<trackerId>` |

## Configuration

Both sketches start with a `MANUAL SETUP` block that has to be filled in before flashing:

- `_ssid`, `_password` — Wi-Fi network the boards join;
- `_broker`, `_mqttPort` — address and port of the MQTT broker (Mosquitto was used in the study);
- anchor: `_anchorId`; tracker: `_trackerId` (`BLE_Tracker_1` … `BLE_Tracker_4` in the dataset).

The tracker additionally holds the per-device calibration constants of the log-distance path-loss model:
`_txPower` (reference RSSI at 1 m, dBm) and `_pathLossExponent` (*n*). The sketch ships with the values of
tracker 4 (−51 dBm, 1.45); the values obtained for all four trackers of the study are listed in
`Dataset/calibration/BLE_Calibration.xlsx`.

## Building

Arduino IDE with the Espressif ESP32 board package (board: *ESP32C6 Dev Module*). Required libraries:
`PubSubClient`, `Adafruit NeoPixel`; the BLE and Wi-Fi libraries ship with the board package.

## Message format

The tracker publishes one message per scan cycle:

```
A1:03.02;A2:04.43;A3:05.30;A4:05.61;A5:04.31;A6:04.85;A7:01.70;A8:02.28
```

distances in metres to anchors A1–A8, `00.00` when an anchor was not heard. The same messages are what the
Matter coordinator stores in its diagnostic-logs buffer, and what the `raw/MQTT` files of the dataset contain
(prefixed with a receiver-side timestamp).
