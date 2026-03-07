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
LABEL = "shoulder_press"

file = open("shoulder_press.csv", "a", newline="")
writer = csv.writer(file)


def notification_handler(sender, data):
    values = struct.unpack("ffffff", data)
    timestamp = time.time()

    row = [timestamp] + list(values) + [LABEL]
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
