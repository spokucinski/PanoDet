# Matter coordinator (`room_air_conditioner_MCIL`)

ESP-IDF / esp-matter firmware of the MCIL coordinator board (ESP32-C6-DevKitC-1). Each coordinator is
physically paired with one tracker (UWB or BLE) and makes that tracker's distance estimates available over
the Matter protocol, next to the existing MQTT path.

The project is derived from the esp-matter `room_air_conditioner` example. The device exposes a single
node with a single `room_air_conditioner` endpoint; the only functional addition is the
**Diagnostic Logs** cluster, which is used as a protocol-compliant transport for distance messages:

- `main/app_main.cpp` — Matter stack start-up, endpoint creation, Wi-Fi and MQTT client set-up;
- `main/app_driver.cpp` — subscription to the tracker's MQTT topic; every received distance message is
  pushed into the log buffer;
- `main/diagnostic-logs-provider-delegate-impl.cpp` / `main/include/…h` — custom `DiagnosticLogsProviderDelegate`:
  circular buffer (five most recent messages), log creation, retrieval, size and clean-up;
- `main/diagnostic_logs/*.log` — placeholder log files inherited from the example (not used for distances).

A Matter controller (CHIP-TOOL in the study, see `../../Utils/matter_log_consolidator`) commissions the
board into the fabric and reads the buffer with `diagnosticlogs retrieve-logs-request`.

## Building

1. Install ESP-IDF and esp-matter following the
   [esp-matter documentation](https://docs.espressif.com/projects/esp-matter/en/latest/esp32/developing.html).
2. `idf.py set-target esp32c6`
3. Set the Wi-Fi credentials and the MQTT broker address (menuconfig / `sdkconfig.defaults`).
4. `idf.py build flash monitor`

Third-party components are not checked in; the IDF component manager resolves them from
`main/idf_component.yml` and `dependencies.lock` during the first build.
