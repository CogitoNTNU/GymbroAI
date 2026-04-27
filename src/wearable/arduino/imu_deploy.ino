/*
 * Exercise Recognition — Arduino Nano 33 BLE Sense
 * ================================================
 * Full int8-quantized TFLite model + BLE communication.
 *
*/

#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

// ─── Include the int8-quantized model header ────────────────────
// This header defines: g_model[], g_model_len, kNumSamples,
// kNumFeatures, kNumClasses, kGestureNames[]
#include "model.h"

// ─── Configuration ──────────────────────────────────────────────
const float accelerationThreshold = 2.5;  // G's to trigger capture
const int numSamples = kNumSamples;       // 119 (from model header)

int samplesRead = numSamples;

// ─── BLE Setup ──────────────────────────────────────────────────
BLEService classService("db6d5260-ae3e-4421-a65c-73ca64cc7d3a");
BLECharacteristic gestureChar("db6d5260-ae3e-4421-a65c-73ca64cc7d3b",
                              BLERead | BLENotify, 32);

// ─── TFLite Micro Setup ─────────────────────────────────────────
// Op resolver with 6 ops:
//   FullyConnected, Relu, Softmax, Reshape — model layers
//   Quantize, Dequantize — added by converter for float32 I/O on int8 model
static tflite::MicroMutableOpResolver<6> tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Tensor arena — 16KB is plenty for this int8 model
// (int8 uses ~4x less working memory than float32)
constexpr int tensorArenaSize = 16 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Placement-new buffer for interpreter (avoids heap fragmentation)
static uint8_t interpreterBuffer[sizeof(tflite::MicroInterpreter)]
    __attribute__((aligned(16)));


void setup() {
  Serial.begin(9600);
  //while (!Serial);

  Serial.println("=== Exercise Recognition ===");
  Serial.print("Classes: ");
  Serial.println(kNumClasses);
  for (int i = 0; i < kNumClasses; i++) {
    Serial.print("  ");
    Serial.print(i);
    Serial.print(": ");
    Serial.println(kGestureNames[i]);
  }

  // ── Initialize IMU ──
  if (!IMU.begin()) {
    Serial.println("ERROR: IMU init failed!");
    while (1);
  }
  Serial.print("Accel rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  // ── Initialize BLE ──
  if (!BLE.begin()) {
    Serial.println("ERROR: BLE init failed!");
    while (1);
  }
  BLE.setLocalName("CogitoIMU");
  BLE.setAdvertisedService(classService);
  classService.addCharacteristic(gestureChar);
  BLE.addService(classService);
  BLE.advertise();
  Serial.println("BLE advertising...");

  // ── Register TFLite ops ──
  // These 6 ops cover the full int8-quantized model:
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddRelu();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddQuantize();
  tflOpsResolver.AddDequantize();  

  // ── Load model ──
  tflModel = tflite::GetModel(g_model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.print("ERROR: Model schema mismatch! Got ");
    Serial.print(tflModel->version());
    Serial.print(", expected ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1);
  }

  // ── Create interpreter ──
  tflInterpreter = new (interpreterBuffer) tflite::MicroInterpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    while (1);
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  // ── Print diagnostics ──
  Serial.print("Arena used: ");
  Serial.print(tflInterpreter->arena_used_bytes());
  Serial.print(" / ");
  Serial.print(tensorArenaSize);
  Serial.println(" bytes");

  Serial.print("Input:  type=");
  Serial.print(tflInputTensor->type);  // 1=float32, 9=int8
  Serial.print(", bytes=");
  Serial.println(tflInputTensor->bytes);

  Serial.print("Output: type=");
  Serial.print(tflOutputTensor->type);
  Serial.print(", bytes=");
  Serial.println(tflOutputTensor->bytes);

  Serial.println("Ready — waiting for motion...\n");
}


void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected: ");
    Serial.println(central.address());

    while (central.connected()) {
      float aX, aY, aZ, gX, gY, gZ;

      // ── Wait for significant motion ──
      while (samplesRead == numSamples) {
        BLE.poll();  // Keep BLE alive while waiting
        if (IMU.accelerationAvailable()) {
          IMU.readAcceleration(aX, aY, aZ);
          float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
          if (aSum >= accelerationThreshold) {
            samplesRead = 0;
            break;
          }
        }
      }

      // ── Collect samples ──
      while (samplesRead < numSamples) {
        // NOTE: No BLE.poll() during data collection to avoid
        // interrupt-driven corruption of the sample buffer
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
          IMU.readAcceleration(aX, aY, aZ);
          IMU.readGyroscope(gX, gY, gZ);

          // Normalize to [0, 1] — same formula as Python training
          int idx = samplesRead * 6;
          tflInputTensor->data.f[idx + 0] = (aX + 4.0f) / 8.0f;
          tflInputTensor->data.f[idx + 1] = (aY + 4.0f) / 8.0f;
          tflInputTensor->data.f[idx + 2] = (aZ + 4.0f) / 8.0f;
          tflInputTensor->data.f[idx + 3] = (gX + 2000.0f) / 4000.0f;
          tflInputTensor->data.f[idx + 4] = (gY + 2000.0f) / 4000.0f;
          tflInputTensor->data.f[idx + 5] = (gZ + 2000.0f) / 4000.0f;

          samplesRead++;
        }
      }

      // ── Run inference (BLE paused) ──
      Serial.println("Invoking...");

      // CRITICAL: Do NOT call BLE.poll() between filling input and
      // reading output. SoftDevice interrupts can corrupt FPU state
      // and tensor arena during inference.
      TfLiteStatus invokeStatus = tflInterpreter->Invoke();

      if (invokeStatus != kTfLiteOk) {
        Serial.println("ERROR: Invoke failed!");
        samplesRead = numSamples;  // Reset for next attempt
        continue;
      }

      // ── Read results ──
      int maxIndex = 0;
      float maxValue = tflOutputTensor->data.f[0];

      // Check for NaN (safety net)
      if (isnan(maxValue)) {
        Serial.println("WARNING: NaN detected in output — skipping");
        samplesRead = numSamples;
        continue;
      }

      for (int i = 1; i < kNumClasses; i++) {
        float val = tflOutputTensor->data.f[i];
        if (!isnan(val) && val > maxValue) {
          maxValue = val;
          maxIndex = i;
        }
      }

      // Print result
      Serial.print(">>> ");
      Serial.print(kGestureNames[maxIndex]);
      Serial.print(" (");
      Serial.print(maxValue * 100.0f, 1);
      Serial.println("%)");

      // ── Send via BLE ──
      char buffer[32];
      snprintf(buffer, sizeof(buffer), "%s|%.1f",
               kGestureNames[maxIndex], maxValue * 100.0f);
      gestureChar.writeValue(buffer);

      // Resume BLE polling
      BLE.poll();
    }

    Serial.println("Disconnected");
  }
}
