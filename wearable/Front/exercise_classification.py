import asyncio
from bleak import BleakScanner, BleakClient
from dotenv import load_dotenv
import os

load_dotenv()
DEVICE_NAME = "CogitoIMU"
GEST_UUID = os.getenv("UUID2")

bicep_curl_counter = 0
shoulder_press_counter = 0


def gest_callback(sender, data):
    data = data.decode()
    gesture, confidence = data.split("|")
    print(f"Gesture: {gesture}, Confidence: {confidence}")
    if gesture == "biceps-curl" and float(confidence) > 0.8:
        global bicep_curl_counter
        bicep_curl_counter += 1
        print(f"Bicep Curl Count: {bicep_curl_counter}")
    elif gesture == "shoulder-press" and float(confidence) > 0.8:
        global shoulder_press_counter
        shoulder_press_counter += 1
        print(f"Shoulder Press Count: {shoulder_press_counter}")

    print(
        f"Current Counts - Bicep Curls: {bicep_curl_counter}, Shoulder Presses: {shoulder_press_counter}"
    )


async def main():
    # Scan for the device
    print("Scanning for devices...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if device is None:
        print(f"Device '{DEVICE_NAME}' not found.")
        return

    # Connect to the device
    async with BleakClient(device) as client:
        print(f"Connected to {DEVICE_NAME}")

        # Start notifications for gesture data
        await client.start_notify(GEST_UUID, gest_callback)

        print("Listening for data... Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Stopping notifications...")

        # Stop notifications
        await client.stop_notify(GEST_UUID)


asyncio.run(main())
