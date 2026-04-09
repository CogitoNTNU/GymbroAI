import asyncio
import struct
import csv
import os
from dotenv import load_dotenv
from bleak import BleakClient, BleakScanner

load_dotenv()
DEVICE_NAME = "CogitoIMU"
CHAR_UUID = os.getenv("UUID")

person = "dj"
exercise = "squat"

folder = "data0\\"
os.makedirs(folder, exist_ok=True)  # lager mappen hvis den ikke finnes

filename = f"{exercise}_{person}.csv"
filepath = os.path.join(folder, filename)

file = open(filepath, "a", newline="")
writer = csv.writer(file)


def notification_handler(sender, data):
    values = struct.unpack("ffffff", data)
    row = list(values)
    writer.writerow(row)
    print(row)


async def main():
    # Scan for the device
    print("Scanning for devices...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if device is None:
        print(f"Device '{DEVICE_NAME}' not found.")
        return

    async with BleakClient(device) as client:
        print(f"Connected to {DEVICE_NAME}")

        csv.writer(file).writerow(["aX", "aY", "aZ", "gX", "gY", "gZ"])

        await client.start_notify(CHAR_UUID, notification_handler)

        print("Recording... Press Ctrl+C to stop.")

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping recording...")

        await client.stop_notify(CHAR_UUID)

    file.close()
    print("File saved.")


asyncio.run(main())
