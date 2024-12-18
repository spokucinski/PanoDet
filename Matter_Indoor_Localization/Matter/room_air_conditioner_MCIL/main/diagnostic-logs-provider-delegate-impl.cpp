/*
 *
 *    Copyright (c) 2024 Project CHIP Authors
 *    All rights reserved.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

#include <diagnostic-logs-provider-delegate-impl.h>
#include <lib/support/SafeInt.h>
#include <esp_log.h>
#include <string>

#if defined(CONFIG_ESP_COREDUMP_ENABLE_TO_FLASH) && defined(CONFIG_ESP_COREDUMP_DATA_FORMAT_ELF)
#include <esp_core_dump.h>
#endif // defined(CONFIG_ESP_COREDUMP_ENABLE_TO_FLASH) && defined(CONFIG_ESP_COREDUMP_DATA_FORMAT_ELF)

using namespace chip;
using namespace chip::app::Clusters::DiagnosticLogs;

static const char *TAG = "### - DiagnosticLogsProvider";

LogProvider LogProvider::sInstance;
LogProvider::CrashLogContext LogProvider::sCrashLogContext;

// Total size allocated for the log buffer
// const size_t totalLogMemorySize = 32768; // 32kB
const size_t totalLogMemorySize = 360; // 5 lines of readings

// Log buffer for in-memory logs.
u_int8_t logBuffer[totalLogMemorySize] = {};
u_int8_t currentLogCount = 0;

// Pointers to the beginning and current end of the log buffer
uint8_t* logStart = logBuffer;
uint8_t* logEnd = logBuffer;

LogProvider::~LogProvider()
{
    for (auto sessionSpan : mSessionContextMap)
    {
        Platform::MemoryFree(sessionSpan.second);
    }
    mSessionContextMap.clear();
}

bool IsValidIntent(IntentEnum intent)
{
    ESP_LOGI(TAG, "Checking log request intent");
    return intent == IntentEnum::kEndUserSupport;
}

// Function to add a log entry
void LogProvider::AddLogEntry(const char* logEntry, size_t logEntryLength) {

    ESP_LOGI(TAG, "Started adding a new log");

    // Calculate the size of the new log entry
    // and ensure the logEntryLength does not exceed the actual length of the logEntry
    size_t newLogEntrySize = std::min(std::strlen(logEntry), logEntryLength);
    ESP_LOGI(TAG, "Calculated newLogEntrySize: %zu", newLogEntrySize);

    // Calculate available space from current logEnd to the end of the buffer
    size_t spaceAtEnd = totalLogMemorySize - (logEnd - logStart);
    ESP_LOGI(TAG, "Calculated spaceAtEnd: %zu", spaceAtEnd);

    // Check if there is enough space at the end of the buffer
    if (newLogEntrySize + 1 > spaceAtEnd) { // +1 for newline or null terminator
        // Not enough space at the end, so wrap around
        if (newLogEntrySize + 1 > totalLogMemorySize) {
            // If the new entry is too large to fit in the buffer, truncate it
            newLogEntrySize = totalLogMemorySize - 1;
        }

        logEnd = logStart;
    }

    // Append the new log entry to the current log
    std::memcpy(logEnd, logEntry, newLogEntrySize);

    // Update the log end pointer
    logEnd += newLogEntrySize;

    // Optionally, add a newline or null terminator if needed
    *logEnd = '\n'; // or '\0' if you prefer null-terminated strings
    logEnd++;

    // Wrap around if necessary
    if (logEnd >= logStart + totalLogMemorySize) {
        logEnd = logStart + (logEnd - (logStart + totalLogMemorySize));
    }
}

void LogProvider::InitializeLogBuffer(){
    ESP_LOGI(TAG, "Log buffer initialization");
    //AddLogEntry("### LOG BUFFER INITIALIZATION ###");
}

CHIP_ERROR LogProvider::GetLogForIntent(IntentEnum intent, MutableByteSpan & outBuffer, Optional<uint64_t> & outTimeStamp,
                                        Optional<uint64_t> & outTimeSinceBoot)
{
    ESP_LOGI(TAG, "Getting log");

    CHIP_ERROR err                 = CHIP_NO_ERROR;
    LogSessionHandle sessionHandle = kInvalidLogSessionHandle;
  
    currentLogCount++;
    ESP_LOGI(TAG, "Increasing log counter. Current state: %d", currentLogCount);
    //std::string text = "### TEST LOG NUMBER: ";
    //text += std::to_string(currentLogCount);
    //AddLogEntry(text.c_str());

    err = StartLogCollection(intent, sessionHandle, outTimeStamp, outTimeSinceBoot);
    VerifyOrReturnError(CHIP_NO_ERROR == err, err, outBuffer.reduce_size(0));

    bool unusedOutIsEndOfLog;
    err = CollectLog(sessionHandle, outBuffer, unusedOutIsEndOfLog);
    VerifyOrReturnError(CHIP_NO_ERROR == err, err, outBuffer.reduce_size(0));

    err = EndLogCollection(sessionHandle);
    VerifyOrReturnError(CHIP_NO_ERROR == err, err, outBuffer.reduce_size(0));

    return CHIP_NO_ERROR;
}

size_t LogProvider::GetSizeForIntent(IntentEnum intent)
{
    ESP_LOGI(TAG, "Calculating log size");

    switch (intent)
    {
        case IntentEnum::kEndUserSupport:
        {
            int logSize = logEnd - logStart;
            ESP_LOGI(TAG, "Calculated log size: %d", logSize);
            return static_cast<size_t>(logEnd - logStart);
        }

        default:
            return 0;
    }
}

CHIP_ERROR LogProvider::PrepareLogContextForIntent(LogContext * context, IntentEnum intent)
{
    ESP_LOGI(TAG, "Initialization of log context");

    context->intent = intent;

    switch (intent)
    {
        case IntentEnum::kEndUserSupport: {

            int calculatedBufferSize = logEnd - logStart;
            ESP_LOGI(TAG, "Log context ByteSpan's size set to: %d", totalLogMemorySize);
            context->EndUserSupport.span =
                ByteSpan(logBuffer, totalLogMemorySize);
        }
        break;

        default:
            return CHIP_ERROR_INVALID_ARGUMENT;
    }

    return CHIP_NO_ERROR;
}

void LogProvider::CleanupLogContextForIntent(LogContext * context)
{
    ESP_LOGI(TAG, "Cleaning of log context");
    
    switch (context->intent)
    {
        case IntentEnum::kEndUserSupport:
            break;

        default:
            break;
    }
}

CHIP_ERROR LogProvider::GetDataForIntent(LogContext * context, MutableByteSpan & outBuffer, bool & outIsEndOfLog)
{
    ESP_LOGI(TAG, "Getting log data");

    switch (context->intent)
    {
        case IntentEnum::kEndUserSupport: {
            auto dataSize = context->EndUserSupport.span.size();
            auto count    = std::min(dataSize, outBuffer.size());

            VerifyOrReturnError(CanCastTo<off_t>(count), CHIP_ERROR_INVALID_ARGUMENT, outBuffer.reduce_size(0));
            ReturnErrorOnFailure(CopySpanToMutableSpan(ByteSpan(context->EndUserSupport.span.data(), count), outBuffer));

            outIsEndOfLog = dataSize == count;
            if (!outIsEndOfLog)
            {
                // reduce the span after reading count bytes
                context->EndUserSupport.span = context->EndUserSupport.span.SubSpan(count);
            }
        }
        break;

        default:
            return CHIP_ERROR_INVALID_ARGUMENT;
        }

    return CHIP_NO_ERROR;
}

CHIP_ERROR LogProvider::StartLogCollection(IntentEnum intent, LogSessionHandle & outHandle, Optional<uint64_t> & outTimeStamp,
                                           Optional<uint64_t> & outTimeSinceBoot)
{
    ESP_LOGI(TAG, "Starting log collection");

    VerifyOrReturnValue(IsValidIntent(intent), CHIP_ERROR_INVALID_ARGUMENT);

    LogContext * context = reinterpret_cast<LogContext *>(Platform::MemoryCalloc(1, sizeof(LogContext)));
    VerifyOrReturnValue(context != nullptr, CHIP_ERROR_NO_MEMORY);

    CHIP_ERROR err = PrepareLogContextForIntent(context, intent);
    VerifyOrReturnError(err == CHIP_NO_ERROR, err, Platform::MemoryFree(context));

    mLogSessionHandle++;
    // If the session handle rolls over to UINT16_MAX which is invalid, reset to 0.
    VerifyOrDo(mLogSessionHandle != kInvalidLogSessionHandle, mLogSessionHandle = 0);

    outHandle                             = mLogSessionHandle;
    mSessionContextMap[mLogSessionHandle] = context;

    return CHIP_NO_ERROR;
}

CHIP_ERROR LogProvider::EndLogCollection(LogSessionHandle sessionHandle)
{
    ESP_LOGI(TAG, "Ending log collection");

    VerifyOrReturnValue(sessionHandle != kInvalidLogSessionHandle, CHIP_ERROR_INVALID_ARGUMENT);
    VerifyOrReturnValue(mSessionContextMap.count(sessionHandle), CHIP_ERROR_INVALID_ARGUMENT);

    LogContext * context = mSessionContextMap[sessionHandle];
    VerifyOrReturnError(context, CHIP_ERROR_INCORRECT_STATE);

    CleanupLogContextForIntent(context);
    Platform::MemoryFree(context);
    mSessionContextMap.erase(sessionHandle);

    return CHIP_NO_ERROR;
}

CHIP_ERROR LogProvider::CollectLog(LogSessionHandle sessionHandle, MutableByteSpan & outBuffer, bool & outIsEndOfLog)
{
    ESP_LOGI(TAG, "Collecting log");
    VerifyOrReturnValue(sessionHandle != kInvalidLogSessionHandle, CHIP_ERROR_INVALID_ARGUMENT);
    VerifyOrReturnValue(mSessionContextMap.count(sessionHandle), CHIP_ERROR_INVALID_ARGUMENT);

    LogContext * context = mSessionContextMap[sessionHandle];
    VerifyOrReturnError(context, CHIP_ERROR_INCORRECT_STATE);

    return GetDataForIntent(context, outBuffer, outIsEndOfLog);
}
