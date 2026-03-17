import asyncio
import os
from dotenv import load_dotenv
from bleak import BleakClient

load_dotenv()
address = os.getenv("MAC_ADRESS")
char_uuid = os.getenv("UUID2")

ADDRESS = address
CHAR_UUID = char_uuid


def notification_handler(sender, data):
    print(f"Notification from {sender}: {data}")


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


asyncio.run(main())
