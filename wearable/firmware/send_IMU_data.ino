#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

// Sett inn dine egne UUID-er her (kan genereres på https://www.uuidgenerator.net/)
BLEService imuService("3ba9f7b5-8e2d-4f12-8338-08e1296dd5fe"); 
BLECharacteristic imuCharacteristic(
  "3ba9f7b5-8e2d-4f12-8338-08e1296dd5ff",
  BLENotify,
  24
);


const float accelerationThreshold = 0;  
const int sampleRate = 50;                
const int captureSeconds = 2;
const int numSamples = sampleRate * captureSeconds;


int samplesRead = numSamples;

void setup() {
  Serial.begin(9600);

  if (!BLE.begin()) {
    while (1);
  }

  if (!IMU.begin()) {
    while (1);
  }

  BLE.setLocalName("CogitoIMU");
  BLE.setAdvertisedService(imuService);

  imuService.addCharacteristic(imuCharacteristic);
  BLE.addService(imuService);

  BLE.advertise();
  Serial.println("Advertising...");
}

void loop() {

  BLEDevice central = BLE.central();

  if (central) {
    Serial.println("Connected");

    while (central.connected()) {

      float ax, ay, az, gx, gy, gz;

      // ===== 1) Vent på significant motion =====
      while (samplesRead == numSamples && central.connected()) {

        if (IMU.accelerationAvailable()) {

          IMU.readAcceleration(ax, ay, az);

          float aSum = fabs(ax) + fabs(ay) + fabs(az);

          if (aSum >= accelerationThreshold) {
            samplesRead = 0;  // start capture
            Serial.println("Motion detected");
            break;
          }
        }
      }

      
      while (samplesRead < numSamples && central.connected()) {

        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {

          IMU.readAcceleration(ax, ay, az);
          IMU.readGyroscope(gx, gy, gz);

          float data[6] = {ax, ay, az, gx, gy, gz};
          imuCharacteristic.writeValue((byte*)data, sizeof(data));

          samplesRead++;

          delay(1000 / sampleRate);  
        }
      }

      
      if (samplesRead == numSamples) {
        Serial.println("Capture complete");
      }
    }

    Serial.println("Disconnected");
  }
}