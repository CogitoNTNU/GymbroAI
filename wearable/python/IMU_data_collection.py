import asyncio
import struct
import csv
import time
import os
from dotenv import load_dotenv
from bleak import BleakClient

load_dotenv()
address = os.getenv("MAC_ADRESS")
char_uuid = os.getenv("UUID")

ADDRESS = address
CHAR_UUID = char_uuid
# Tricep extension= 0 Shoulder press = 1, Bicep curl = 2, Squat = 3, Rows = 4
LABEL = 3

recording_number = 1
person = "test"
exercise = "squat"

folder = "data\\" + exercise
os.makedirs(folder, exist_ok=True)  # lager mappen hvis den ikke finnes

filename = f"{exercise}_{person}_{recording_number}.csv"
filepath = os.path.join(folder, filename)

file = open(filepath, "a", newline="")
writer = csv.writer(file)


def notification_handler(sender, data):
    values = struct.unpack("ffffff", data)
    timestamp = time.time()

    row = [timestamp] + list(values) + [LABEL] + [person]
    writer.writerow(row)
    print(row)


async def main():
    async with BleakClient(ADDRESS) as client:
        print("Connected:", client.is_connected)

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
