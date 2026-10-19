# Utilities

Host-side scripts used during the experiments.

## `matter_log_consolidator/`

- `matterAnalyzer.py` — polls all Matter coordinators through CHIP-TOOL. For each tracker it runs
  `diagnosticlogs retrieve-logs-request` in a separate CHIP-TOOL instance (one commissioned device per
  storage directory), parses the returned hex-encoded log into distance measurements, stores them per
  retrieval tick as `tracker_<k>.txt` and optionally forwards them to Azure Event Grid. The files under
  `Dataset/raw/Matter` were produced with this script.
- `mqttAnalyzer.py` — subscribes to the trackers' MQTT topics, time-stamps every message and forwards it to
  Azure Event Grid. The files under `Dataset/raw/MQTT` were produced with this script.
- `distanceMeasurement.py`, `error.py` — data classes.
- `secrets.example.py` — configuration template. Copy it to `secrets.py` (ignored by git) and fill in the
  CHIP-TOOL path and storage directory, the number of trackers, the output path and, if forwarding is used,
  the Event Grid endpoint and key.

Requirements: Python 3.10+, `paho-mqtt`, `azure-eventgrid`, a built CHIP-TOOL binary.

## `multilateration/`

`multilateration.py` — least-squares estimation of a 3D tracker position from anchor coordinates and
anchor–tracker distances (`numpy`, `scipy`). Position estimation is not part of the published study; the
script is provided as a starting point for working with the dataset.
