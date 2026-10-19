# Matter-Coordinated Indoor Localization (MCIL)

Companion repository for the study *An Indoor Positioning Method for IoT Devices Using Matter Protocol and Digital Twin Modeling* (S. Pokuciński, K. Hanzel, K. Marszałek, W. Ding, D. Mrozek).

MCIL extends an existing UWB-based indoor distance estimation system with a BLE-based subsystem and a custom
Matter coordinator board, so that distance estimates from both radio technologies are exposed simultaneously
over the legacy MQTT pipeline and over the Matter protocol. Ground truth for the experiments is taken from
three sources: a laser rangefinder, the interactive measurement tool of a Matterport virtual tour (3D Walk),
and a full volumetric 3D model of the room (3D Model).

## Repository contents

| Folder | Content |
|---|---|
| `BLE/BLE_Anchor` | Arduino firmware for the BLE anchor (ESP32-C6-DevKitC-1): BLE advertiser, Wi-Fi + MQTT status reporting |
| `BLE/BLE_Tracker` | Arduino firmware for the BLE tracker (ESP32-C6-DevKitC-1): BLE scanner, RSSI-based distance estimation, MQTT publishing |
| `Matter/room_air_conditioner_MCIL` | ESP-IDF / esp-matter project of the Matter coordinator: a `room_air_conditioner` endpoint whose `diagnostic_logs` cluster stores incoming MQTT distance messages in a circular buffer |
| `Utils/matter_log_consolidator` | Python scripts used during the experiments: CHIP-TOOL log retrieval from all coordinators (`matterAnalyzer.py`) and MQTT capture with forwarding to Azure Event Grid (`mqttAnalyzer.py`) |
| `Utils/multilateration` | Least-squares 3D multilateration helper |
| `Dataset` | Raw and processed measurements of Test Case 1 (E1–E17, 264 series), ground truth, anchor positions, BLE calibration and the aggregated result tables — see `Dataset/README.md` |

The UWB anchor and tracker firmware is part of the earlier localization system the framework builds upon and is described in the publications referenced in the article; it is not distributed here.

## Hardware

- UWB anchors and trackers: Makerfabs ESP32 UWB (ESP32 + Decawave DW3000)
- BLE anchors and trackers, Matter coordinator: Espressif ESP32-C6-DevKitC-1
- Matter controller: Raspberry Pi 4B running CHIP-TOOL
- Each physical anchor unit combines one UWB and one BLE anchor module with its own power bank; each tracker is paired with its own Matter coordinator board.

## Building the firmware

- `BLE/*`: open the sketch in the Arduino IDE with the ESP32 board package, fill in the Wi-Fi and MQTT broker fields at the top of the file, flash.
- `Matter/room_air_conditioner_MCIL`: standard esp-matter project; set up ESP-IDF and esp-matter, run `idf.py set-target esp32c6`, `idf.py build flash`. Dependencies are resolved from `main/idf_component.yml` / `dependencies.lock` by the IDF component manager.
- `Utils/matter_log_consolidator`: fill in `secrets.py` (CHIP-TOOL path and storage directory, number of trackers, output path, optional Event Grid endpoint/key) before running.
