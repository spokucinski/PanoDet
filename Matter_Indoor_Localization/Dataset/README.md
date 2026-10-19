# MCIL dataset — Test Case 1

Measurement dataset collected with the Matter-Coordinated Indoor Localization (MCIL) framework
in a single furnished room (brick office building, residential arrangement).
17 experiment configurations (E1–E17), each repeated 3 times (vol 1–3), 8 anchors,
up to 4 UWB and 4 BLE trackers, readings collected simultaneously over MQTT and over the
Matter protocol — 264 measurement series in total.

## Layout

| Path | Content |
|---|---|
| `experiments.csv` | Configuration of E1–E17: technology, static/dynamic, placement, acquisition time, repetitions, number of series |
| `anchors.csv` | Anchor positions (X, Y, Z in cm, from the 3D model) and anchor elevation above the floor from the three ground-truth sources |
| `ground_truth/gt_ground_level_h0.csv` | Reference tracker–anchor distances (cm) for measurement points P1–P10 at elevation 0 cm, from laser rangefinder, 3D Walk and 3D Model |
| `ground_truth/gt_tripod_level_h150.csv` | Same for the tripod elevation (150 cm); points P1, P3, P5, P7, P10 |
| `calibration/BLE_Calibration.xlsx` | RSSI reference (1 m) and path-loss exponent (3 m) calibration samples for BLE trackers 1–4, with per-tracker means and standard deviations |
| `raw/MQTT/` | Unprocessed MQTT logs, one folder per experiment and repetition (`EXP{n}_vol{v}_<timestamp>`), one file per tracker |
| `raw/Matter/` | Unprocessed Matter logs retrieved with CHIP-TOOL, one folder per experiment and repetition (`EXP{n}_vol{v}_matter`), one sub-folder per retrieval tick (`<timestamp>`), one file per tracker |
| `processed/Exp_NN.xlsx` | Per-experiment workbooks: one sheet per repetition and protocol (`vol v, MQTT` / `vol v, Matter`), readings aligned per tracker and anchor |
| `summary/Summary.xlsx` | Aggregated statistics: per experiment/tracker/protocol errors against each ground-truth source (`data`), global and protocol-wise summaries for ground level (`*H0`) and tripod level (`*H150`), and the bucketing input sheets |
| `bucketing/*.csv` | Per-series statistics used for distance bucketing, split by protocol, technology and tracker elevation (`GL` ground level, `TL` tripod level) |
| `bucketing/out/` | Bucketing results per split: grouped statistics (`.csv`, `.tex`) and the mean-absolute-error-by-range charts (`.png`) |

Tracker numbering: trackers 1–4 are UWB trackers 1–4, trackers 5–8 are BLE trackers 1–4.
Measurement points P1–P10 are the floor markers of the test room; see `experiments.csv` for which tracker stood where.

## File formats

### `raw/MQTT/.../{UWB,BLE}_Tracking_{UWB,BLE}_Tracker_{k}.txt`
One line per message published by the tracker:

```
2024-07-20T09:05:49.187892: A1:03.02;A2:04.43;A3:05.30;A4:05.61;A5:04.31;A6:04.85;A7:01.70;A8:02.28
```

ISO timestamp (receiver clock), then the estimated distance in metres to anchors A1–A8.
`00.00` means the anchor was not heard in that cycle; `99.99` is the out-of-range marker.

### `raw/Matter/.../<timestamp>/tracker_{k}.txt`
Output of one `diagnosticlogs retrieve-logs-request` issued to the Matter coordinator of tracker *k*:

```
Tracker 1 collected 40 measurements.
Anchor ID: 1, Distance: 3.00, Exec Time: 1.04s
...
```

Distances in metres; `Exec Time` is the CHIP-TOOL round-trip time. A tracker whose coordinator
ring buffer was empty at retrieval time reports `collected 0 measurements` and an `Error: NoLogs` line.
The raw files are left exactly as recorded, including such empty retrievals and device outages
(e.g. BLE tracker 1 over Matter during E1).

### `processed/Exp_NN.xlsx`
Columns per tracker: `Timestamp`, `A1` … `A8`. Where present, `An PF` is the pre-filtered value
(zeros and `99.99` removed) and `An F` the filtered value (within one standard deviation of the mean).

### `bucketing/*.csv` (semicolon-separated)
`Experiment; Tracker; Technology; Protocol; Anchor; MeanUnfilteredValue; StandardDeviation; MeanFilteredValue; GT; ErrorM; ErrorPercent; GTType; Vol`
with `GTType` ∈ {`Laser`, `Walk`, `Model`} and distances in metres.

### `ground_truth/*.csv`, `anchors.csv`
Distances and coordinates in centimetres. `walk3d_cm` is empty where the 3D Walk tool could not place
a point (anchor A7, floor obstructed by furniture).
