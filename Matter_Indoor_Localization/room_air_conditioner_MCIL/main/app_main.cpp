/*
   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_wifi.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "protocol_examples_common.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include "lwip/sockets.h"
#include "lwip/dns.h"
#include "lwip/netdb.h"
#include "esp_log.h"
#include "mqtt_client.h"
#include <esp_err.h>
#include <nvs_flash.h>
#include <esp_matter.h>
#include <esp_matter_console.h>
#include <esp_matter_ota.h>
#include <common_macros.h>
#include <app_priv.h>
#include <app_reset.h>
#if CHIP_DEVICE_CONFIG_ENABLE_THREAD
#include <platform/ESP32/OpenthreadLauncher.h>
#endif
#include <app/server/CommissioningWindowManager.h>
#include <app/server/Server.h>
#include <diagnostic-logs-provider-delegate-impl.h>
#include <app/clusters/diagnostic-logs-server/diagnostic-logs-server.h>

// ##### ##### ##### MANUAL CONFIG ##### ##### ##### 
static const char *TAG = "##### AirConditionerMCIL #####";
static const char *TRACKER_NAME = "UWB_Tracker_1";
static const char *TRACKER_TOPIC = "UWB_Tracking/UWB_Tracker_1";
static const char *STATUS_TOPIC = "UWB_Status";
static const char *MQTT_URI = "mqtt://192.168.199.197:1885";

uint16_t room_air_conditioner_endpoint_id = 0;

using namespace esp_matter;
using namespace esp_matter::attribute;
using namespace esp_matter::endpoint;
using namespace chip::app::Clusters;

constexpr auto k_timeout_seconds = 300;

cluster_t* myCluster = NULL;

void delay_ms(int milliseconds) {
    vTaskDelay(pdMS_TO_TICKS(milliseconds));
}

static void log_error_if_nonzero(const char *message, int error_code)
{
    if (error_code != 0) {
        ESP_LOGE(TAG, "Last error %s: 0x%x", message, error_code);
    }
}

/*
 * @brief Event handler registered to receive MQTT events
 *
 *  This function is called by the MQTT client event loop.
 *
 * @param handler_args user data registered to the event.
 * @param base Event base for the handler(always MQTT Base in this example).
 * @param event_id The id for the received event.
 * @param event_data The data for the event, esp_mqtt_event_handle_t.
 */
using namespace chip::app::Clusters::DiagnosticLogs;
static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
    ESP_LOGI(TAG, "MQQT Event handler called!");

    esp_mqtt_event_handle_t event = (esp_mqtt_event_handle_t)event_data;
    esp_mqtt_client_handle_t client = event->client;

    // NOTES:
    // Sample publish of message
    // client, topic, data (message), length (0 - autocalculated), Qos (1), Retain (0)
    // msg_id = esp_mqtt_client_publish(client, "/topic/qos1", "data_3", 0, 1, 0);
    //  
    // Sample subscribe
    // client, topic, QoS (1)
    // msg_id = esp_mqtt_client_subscribe(client, "/topic/qos1", 1);

    switch ((esp_mqtt_event_id_t)event_id) {
        
        case MQTT_EVENT_CONNECTED:
            {
                ESP_LOGI(TAG, "MQTT_EVENT_CONNECTED");
                
                ESP_LOGI(TAG, "Broadcasting connected");
                char messageBuffer[32];
                std::snprintf(messageBuffer, sizeof(messageBuffer), "%s CONNECTED!", TRACKER_NAME);
                esp_mqtt_client_publish(client, STATUS_TOPIC, messageBuffer, 0, 1, 0);


                ESP_LOGI(TAG, "Subscribing to tracking topic");
                esp_mqtt_client_subscribe(client, TRACKER_TOPIC, 1);
            }
            break;

        case MQTT_EVENT_DISCONNECTED:
            ESP_LOGI(TAG, "MQTT_EVENT_DISCONNECTED");
            break;

        case MQTT_EVENT_SUBSCRIBED:
            ESP_LOGI(TAG, "Tracker Subscribed!");
            break;

        case MQTT_EVENT_UNSUBSCRIBED:
            ESP_LOGI(TAG, "MQTT_EVENT_UNSUBSCRIBED, msg_id=%d", event->msg_id);
            break;

        case MQTT_EVENT_PUBLISHED:
            ESP_LOGI(TAG, "MQTT_EVENT_PUBLISHED, msg_id=%d", event->msg_id);
            break;

        case MQTT_EVENT_DATA:
            {
                ESP_LOGI(TAG, "MQTT_EVENT_DATA, Got message, printing!");
                printf("Topic = %.*s\r\n", event->topic_len, event->topic);
                printf("Message = %.*s\r\n", event->data_len, event->data);
                ESP_LOGI(TAG, "Processing MQTT message to Log entry");
                auto & logProvider = chip::app::Clusters::DiagnosticLogs::LogProvider::GetInstance();
                logProvider.AddLogEntry(event->data, event->data_len);
            }
            break;

        case MQTT_EVENT_ERROR:
            {
                ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
                if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                    log_error_if_nonzero("reported from esp-tls", event->error_handle->esp_tls_last_esp_err);
                    log_error_if_nonzero("reported from tls stack", event->error_handle->esp_tls_stack_err);
                    log_error_if_nonzero("captured as transport's socket errno",  event->error_handle->esp_transport_sock_errno);
                    ESP_LOGI(TAG, "Last errno string (%s)", strerror(event->error_handle->esp_transport_sock_errno));

                }
            }
            break;

        default:
            ESP_LOGI(TAG, "Other event id:%d", event->event_id);
            break;
    }
}

static void app_event_cb(const ChipDeviceEvent *event, intptr_t arg)
{
    switch (event->Type) {
        case chip::DeviceLayer::DeviceEventType::kInterfaceIpAddressChanged:
            ESP_LOGI(TAG, "Interface IP Address changed");
            break;

        case chip::DeviceLayer::DeviceEventType::kCommissioningComplete:
            ESP_LOGI(TAG, "Commissioning complete");
            break;

        case chip::DeviceLayer::DeviceEventType::kFailSafeTimerExpired:
            ESP_LOGI(TAG, "Commissioning failed, fail safe timer expired");
            break;

        case chip::DeviceLayer::DeviceEventType::kCommissioningSessionStarted:
            ESP_LOGI(TAG, "Commissioning session started");
            break;

        case chip::DeviceLayer::DeviceEventType::kCommissioningSessionStopped:
            ESP_LOGI(TAG, "Commissioning session stopped");
            break;

        case chip::DeviceLayer::DeviceEventType::kCommissioningWindowOpened:
            ESP_LOGI(TAG, "Commissioning window opened");
            break;

        case chip::DeviceLayer::DeviceEventType::kCommissioningWindowClosed:
            ESP_LOGI(TAG, "Commissioning window closed");
            break;

        case chip::DeviceLayer::DeviceEventType::kFabricRemoved:
            {
                ESP_LOGI(TAG, "Fabric removed successfully");
                if (chip::Server::GetInstance().GetFabricTable().FabricCount() == 0)
                {
                    chip::CommissioningWindowManager & commissionMgr = chip::Server::GetInstance().GetCommissioningWindowManager();
                    constexpr auto kTimeoutSeconds = chip::System::Clock::Seconds16(k_timeout_seconds);
                    if (!commissionMgr.IsCommissioningWindowOpen())
                    {
                        /* After removing last fabric, this example does not remove the Wi-Fi credentials
                        * and still has IP connectivity so, only advertising on DNS-SD.
                        */
                        CHIP_ERROR err = commissionMgr.OpenBasicCommissioningWindow(kTimeoutSeconds,
                                                        chip::CommissioningWindowAdvertisement::kDnssdOnly);
                        if (err != CHIP_NO_ERROR)
                        {
                            ESP_LOGE(TAG, "Failed to open commissioning window, err:%" CHIP_ERROR_FORMAT, err.Format());
                        }
                    }
                }
            break;
            }

        case chip::DeviceLayer::DeviceEventType::kFabricWillBeRemoved:
            ESP_LOGI(TAG, "Fabric will be removed");
            break;

        case chip::DeviceLayer::DeviceEventType::kFabricUpdated:
            ESP_LOGI(TAG, "Fabric is updated");
            break;

        case chip::DeviceLayer::DeviceEventType::kFabricCommitted:
            ESP_LOGI(TAG, "Fabric is committed");
            break;
        default:
            break;
    }
}

// This callback is invoked when clients interact with the Identify Cluster.
// In the callback implementation, an endpoint can identify itself. (e.g., by flashing an LED or light).
static esp_err_t app_identification_cb(identification::callback_type_t type, uint16_t endpoint_id, uint8_t effect_id,
                                       uint8_t effect_variant, void *priv_data)
{
    ESP_LOGI(TAG, "Identification callback: type: %u, effect: %u, variant: %u", type, effect_id, effect_variant);
    return ESP_OK;
}

// This callback is called for every attribute update. The callback implementation shall
// handle the desired attributes and return an appropriate error code. If the attribute
// is not of your interest, please do not return an error code and strictly return ESP_OK.
static esp_err_t app_attribute_update_cb(attribute::callback_type_t type, uint16_t endpoint_id, uint32_t cluster_id,
                                         uint32_t attribute_id, esp_matter_attr_val_t *val, void *priv_data)
{
    esp_err_t err = ESP_OK;

    if (type == PRE_UPDATE) {
        /* Driver update */
        app_driver_handle_t driver_handle = (app_driver_handle_t)priv_data;
        err = app_driver_attribute_update(driver_handle, endpoint_id, cluster_id, attribute_id, val);
    }

    return err;
}

using namespace chip::app::Clusters::DiagnosticLogs;
void emberAfDiagnosticLogsClusterInitCallback(chip::EndpointId endpoint)
{
    ESP_LOGI(TAG, "Setting Diagnostic Logs Provider Delegate");
    auto & logProvider = LogProvider::GetInstance();
    DiagnosticLogsServer::Instance().SetDiagnosticLogsProviderDelegate(endpoint, &logProvider);
    
    ESP_LOGI(TAG, "Call for initialize log buffer");
    logProvider.InitializeLogBuffer();
}

extern "C" void app_main()
{
    ESP_LOGI(TAG, "Starting app_main()");
    
    ESP_LOGI(TAG, "ESP NVS Layer Initialization");
    /* Initialize the ESP NVS layer */
    nvs_flash_init();

    ESP_LOGI(TAG, "App Driver Initialization");
    /* Initialize driver */
    app_driver_handle_t room_air_conditioner_handle = app_driver_room_air_conditioner_init();
    app_driver_handle_t button_handle = app_driver_button_init();
    app_reset_button_register(button_handle);

    ESP_LOGI(TAG, "Creating Matter Root Node - Endpoint 0");
    /* Create a Matter node and add the mandatory Root Node device type on endpoint 0 */
    node::config_t node_config;
    node_t *node = node::create(&node_config, app_attribute_update_cb, app_identification_cb);
    ABORT_APP_ON_FAILURE(node != nullptr, ESP_LOGE(TAG, "Failed to create Matter node"));

    ESP_LOGI(TAG, "Creating Matter room_air_conditioner Endpoint");
    room_air_conditioner::config_t room_air_conditioner_config;
    room_air_conditioner_config.on_off.on_off = DEFAULT_POWER;
    endpoint_t *endpoint = room_air_conditioner::create(node, &room_air_conditioner_config, ENDPOINT_FLAG_NONE, room_air_conditioner_handle);
    ABORT_APP_ON_FAILURE(endpoint != nullptr, ESP_LOGE(TAG, "Failed to create room air conditioner endpoint"));
    room_air_conditioner_endpoint_id = endpoint::get_id(endpoint);
    ESP_LOGI(TAG, "Room Air Conditioner created with endpoint_id %d", room_air_conditioner_endpoint_id);

    ESP_LOGI(TAG, "Creating Matter diagnostic_logs Cluster in room_air_conditioner Endpoint");
    esp_matter::cluster::diagnostic_logs::config_t logsConfig;
    myCluster = esp_matter::cluster::diagnostic_logs::create(endpoint, &logsConfig, CLUSTER_FLAG_SERVER);

#if CHIP_DEVICE_CONFIG_ENABLE_THREAD
    ESP_LOGI(TAG, "CHIP_DEVICE_CONFIG_ENABLE_THREAD = True, Enabling Thread");
    /* Set OpenThread platform config */
    esp_openthread_platform_config_t config = {
        .radio_config = ESP_OPENTHREAD_DEFAULT_RADIO_CONFIG(),
        .host_config = ESP_OPENTHREAD_DEFAULT_HOST_CONFIG(),
        .port_config = ESP_OPENTHREAD_DEFAULT_PORT_CONFIG(),
    };
    set_openthread_platform_config(&config);
#endif

    /* Matter start */
    ESP_LOGI(TAG, "Starting Matter");
    esp_err_t err = ESP_OK;
    err = esp_matter::start(app_event_cb);
    ABORT_APP_ON_FAILURE(err == ESP_OK, ESP_LOGE(TAG, "Failed to start Matter, err:%d", err));
    ESP_LOGI(TAG, "Matter started successfully!");

    ESP_LOGI(TAG, "Starting App Driver with defaults");
    /* Starting driver with default values */
    app_driver_room_air_conditioner_set_defaults(room_air_conditioner_endpoint_id);

#if CONFIG_ENABLE_CHIP_SHELL
    ESP_LOGI(TAG, "CONFIG_ENABLE_CHIP_SHELL = True, Enabling CHIP Shell");
    esp_matter::console::diagnostics_register_commands();
    esp_matter::console::wifi_register_commands();
    esp_matter::console::init();
#endif

    ESP_LOGI(TAG, "Starting MQTT communication");

    esp_mqtt_client_config_t mqtt_cfg = 
    {
        .broker = {
            .address = {
                .uri = MQTT_URI
            }
        }
    };

    ESP_LOGI(TAG, "MQTT Config structure ready");
    ESP_LOGI(TAG, "Delaying for Wifi initialization");
    delay_ms(20000);

    ESP_LOGI(TAG, "MQTT Client initialization");
    esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);

    ESP_LOGI(TAG, "MQTT Event handler registering");
    /* The last argument may be used to pass data to the event handler, in this example mqtt_event_handler */
    esp_mqtt_client_register_event(client, MQTT_EVENT_ANY, mqtt_event_handler, NULL);

    ESP_LOGI(TAG, "MQTT Starting client");
    esp_mqtt_client_start(client);
}