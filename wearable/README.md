# Wearable — Exercise Recognition

IMU-based exercise classification running on an Arduino Nano 33 BLE Sense. The device detects motion, runs on-device inference using a TensorFlow Lite Micro model, and broadcasts the recognised exercise and confidence score over BLE.

Recognised exercises: **bicep curl**, **shoulder press**, **row**

______________________________________________________________________

## Project structure

```
wearable/
├── arduino/
│   ├── imu_stream.ino          # Data collection — streams raw IMU over BLE
│   ├── imu_deploy.ino          # Inference + BLE output
│   └── model.h                 # Pre-trained quantized model (ready to flash)
├── data_collection/
│   ├── collect_data.py         # Reads BLE stream, saves labelled CSVs
│   └── data/
│       ├── rows.csv
│       ├── shoulder_press.csv
│       └── bicep_curl.csv
├── training/
│   └── train_model.ipynb       # Google Colab notebook
├── models/
│   ├── model.h5                # Trained Keras model
│   └── model_quantized.tflite  # Quantized TFLite model
└── visualisation/
    ├── receive_results.py      # Receives BLE output, writes results.json
    ├── index.html              # Live results webpage
    └── results.json            # Written at runtime, not tracked in git
```

______________________________________________________________________

## Arduino libraries

Install these via the Arduino Library Manager before opening any sketch:

- `Arduino_BMI270_BMM150`
- `ArduinoBLE`
- `Arduino_TensorFlowLite`

______________________________________________________________________

## Quick start — flash the pre-trained model

A finished `model.h` is already included in `arduino/`. If you just want to try the system without collecting data or training:

1. Open `arduino/imu_deploy.ino` in the Arduino IDE
1. Make sure `model.h` is in the same folder
1. Flash to the board and follow [Step 3](#step-3--deploy-to-arduino) below

______________________________________________________________________

## Step 1 — Collect training data

### 1.1 Flash the streaming sketch

Open `arduino/imu_stream.ino` and flash it to the board. It advertises over BLE and streams raw IMU readings in the format:

```
aX,aY,aZ,gX,gY,gZ
```

### 1.2 Record a session

```bash
python data_collection/collect_data.py --label bicep_curl
```

The script uses `bleak` to connect to the Arduino over BLE and appends incoming rows to `data_collection/data/<label>.csv` until you press `Ctrl+C`. Repeat for each exercise label: `bicep_curl`, `shoulder_press`, `rows`. Aim for at least 50–100 repetitions per exercise.

> **Tip:** keep the device orientation consistent between data collection and deployment — the model is sensitive to how the board is mounted.

______________________________________________________________________

## Step 2 — Train the model

Open the notebook in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CogitoNTNU/GymbroAI/blob/main/src/wearable/training/train_model.ipynb)

Upload the three CSV files from `data_collection/data/` when prompted. The notebook will:

1. Window the data into 119-sample segments (119 × 6 = 714 features)
1. Normalise each axis
1. Train a fully connected network with a Softmax output
1. Apply dynamic range quantization
1. Export `model.h5`, `model_quantized.tflite`, and `model.h`

Download the outputs and place them in:

```
models/model.h5
models/model_quantized.tflite
arduino/model.h          ← replace the pre-trained version
```

______________________________________________________________________

## Step 3 — Deploy to Arduino

Flash `arduino/imu_deploy.ino` to the board. Open the Serial Monitor at 9600 baud to verify:

```
Start
Number of exercises: 3
BLE OK
Advertising..
Arena used bytes: 3952
```

Perform a rep and you should see:

```
Start invoke
Invoke ended
Gesture: bicep_curl (94.2% confidence)
```

**BLE output format:** `exercise-name|confidence` — for example `bicep_curl|94.2`

| Field               | Value                                  |
| ------------------- | -------------------------------------- |
| Device name         | `CogitoIMU`                            |
| Service UUID        | `db6d5260-ae3e-4421-a65c-73ca64cc7d3a` |
| Characteristic UUID | `db6d5260-ae3e-4421-a65c-73ca64cc7d3b` |

______________________________________________________________________

## Step 4 — Visualise results

```bash
python visualisation/receive_results.py
```

The script uses `bleak` to scan for a BLE device named `CogitoIMU`, subscribes to the characteristic, and writes each result to `visualisation/results.json`. Open `visualisation/index.html` in a browser to see a live view — no server needed.

______________________________________________________________________

## Model details

| Property             | Value                                          |
| -------------------- | ---------------------------------------------- |
| Input shape          | (1, 714) — 119 samples × 6 features, flattened |
| Output shape         | (1, 3) — one probability per exercise          |
| Quantized model size | ~41 KB                                         |
| Tensor arena         | 16 KB                                          |
| Quantization         | Dynamic range                                  |

**Normalisation:**

```
accelerometer:  (value + 4.0) / 8.0
gyroscope:      (value + 2000.0) / 4000.0
```

______________________________________________________________________

## Troubleshooting

**`AllocateTensors()` fails** — increase `tensorArenaSize` in `imu_deploy.ino` in 8 KB steps until it succeeds.

**No serial output after flashing** — confirm the Serial Monitor baud rate is 9600. If the board appears to hang, double-press the reset button to enter bootloader mode and reflash.

**BLE device not found** — confirm the Arduino is advertising (`Advertising..` in serial output). On Linux you may need to run the script with `sudo` or add your user to the `bluetooth` group.
