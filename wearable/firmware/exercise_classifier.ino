#include <ArduinoBLE.h>
#include "Arduino_BMI270_BMM150.h"


#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

//Setting up BLE
BLEService classService("12345678-1234-1234-1234-1234567890ab");

BLECharacteristic gestureChar("12345678-1234-1234-1234-1234567890ac",
                              BLERead | BLENotify, 24);

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 48 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

static uint8_t interpreterBuffer[sizeof(tflite::MicroInterpreter)] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* GESTURES[] = {
  "biceps-curl",
  "shoulder-pres"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))


void setup() {
  Serial.begin(9600);
  while(!Serial);
  Serial.println("Start");
  Serial.print("Number of exercises: ");
  Serial.println(NUM_GESTURES);

  
  

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  if (!BLE.begin()) {
    Serial.println("Failed to initialize BLE");
      while (1);
  }
  
  Serial.println("BLE OK");
  
  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  //Start advertising
  BLE.setLocalName("CogitoIMU");
  BLE.setAdvertisedService(classService);

  classService.addCharacteristic(gestureChar); 
  BLE.addService(classService);

  BLE.advertise();
  Serial.println("Advertising..");
  
  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
 

  // Create an interpreter to run the model
  //tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);
  tflInterpreter = new(interpreterBuffer) tflite::MicroInterpreter(
    tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors
  //tflInterpreter->AllocateTensors();

  if (tflInterpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed!");
    while (1);
  }

  Serial.print("Arena used bytes: ");
  Serial.println(tflInterpreter->arena_used_bytes());
  Serial.print("Arena total bytes: ");
  Serial.println(tensorArenaSize);

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.print("Output size: ");
  Serial.println(tflOutputTensor->bytes);
}

void loop() {
  BLEDevice central = BLE.central();
  
  if (central) {
    Serial.println("Connected to central");

    while(central.connected()) {

     
      
      float aX, aY, aZ, gX, gY, gZ;
    
      // wait for significant motion
      while (samplesRead == numSamples) {
        BLE.poll();
        if (IMU.accelerationAvailable()) {
          // read the acceleration data
          IMU.readAcceleration(aX, aY, aZ);
    
          // sum up the absolutes
          float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
          //Serial.print("aSUM: ");
          //Serial.println(aSum);
          // check if it's above the threshold
          if (aSum >= accelerationThreshold) {
            // reset the sample read count
            samplesRead = 0;
            break;
          }
        }
      }
    
      // check if the all the required samples have been read since
      // the last time the significant motion was detected
      while (samplesRead < numSamples) {
        BLE.poll();
        // check if new acceleration AND gyroscope data is available
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
          // read the acceleration and gyroscope data
          IMU.readAcceleration(aX, aY, aZ);
          IMU.readGyroscope(gX, gY, gZ);
    
          // normalize the IMU data between 0 to 1 and store in the model's
          // input tensor
          tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
          tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
          tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
          tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
          tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
          tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;
    
          samplesRead++;
    
          if (samplesRead == numSamples) {
            Serial.println("Running inferencing");
            // Run inferencing
            TfLiteStatus invokeStatus = tflInterpreter->Invoke();
            if (invokeStatus != kTfLiteOk) {
              Serial.println("Invoke failed!");
              while (1);
              return;
            }

            
    
            int maxIndex = 0;
            float maxValue = tflOutputTensor->data.f[0];
            for (int i = 1; i < NUM_GESTURES; i++){
              if (tflOutputTensor->data.f[i] > maxValue){
                maxValue = tflOutputTensor->data.f[i];
                maxIndex = i;
              }
            }
            Serial.print("Gesture: ");
            Serial.print(GESTURES[maxIndex]);
            Serial.print(" (");
            Serial.print(maxValue * 100, 1);
            Serial.println("% confidence)");
            
            
            char buffer[32];
            snprintf(buffer, sizeof(buffer), "%s|%.1f", GESTURES[maxIndex], maxValue * 100);

            gestureChar.writeValue(buffer);
          }
        }
      }
    }
  }
}

