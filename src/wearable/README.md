# Wearable — Treningsgjenkjenning

Klassifiserer treningsøvelser i sanntid med IMU-data på Arduino Nano 33 BLE Sense. Sender resultat over BLE.

Støttede øvelser: **bicep curl**, **shoulder press**, **rows**

______________________________________________________________________

## Arduino-biblioteker

Installer via Arduino Library Manager:

- `Arduino_BMI270_BMM150`
- `ArduinoBLE`
- `Arduino_TensorFlowLite`

______________________________________________________________________

## Rask start

En ferdig `model.h` ligger i `arduino/`. Vil du bare teste systemet, hopp rett til [steg 3](#steg-3--deploy).

______________________________________________________________________

## Steg 1 — Samle treningsdata

Flash `arduino/imu_stream.ino` til brettet. Det streamer rådata over BLE.

Start innsamling:

```bash
python data_collection/collect_data.py --label bicep_curl
```

Trykk `Ctrl+C` for å stoppe. Gjenta for `shoulder_press` og `rows`. Sikt på 50–100 repetisjoner per øvelse.

______________________________________________________________________

## Steg 2 — Tren modellen

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/CogitoNTNU/GymbroAI/blob/BLE/GymBro_AI.ipynb)

Last opp CSV-filene fra `data_collection/data/` når notatboken ber om det. Last ned og legg outputfiler her:

```
models/model.h5
models/model_quantized.tflite
arduino/model.h
```

______________________________________________________________________

## Steg 3 — Deploy

Flash `arduino/imu_deploy.ino`. Åpne Serial Monitor (9600 baud) og sjekk at du ser `Advertising..`. Utfør en repetisjon — du skal se noe slikt:

```
Gesture: bicep_curl (94.2% confidence)
```

______________________________________________________________________

## Steg 4 — Visualisering

```bash
python visualisation/receive_results.py
```

Åpne `visualisation/index.html` i en nettleser for sanntidsvisning.
