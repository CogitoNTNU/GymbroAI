import asyncio
from bleak import BleakScanner, BleakClient

DEVICE_NAME = "CogitoIMU"


async def main():
    print("Scanning...")
    devices = await BleakScanner.discover()

    device = None
    for d in devices:
        print(d)
        if d.name == DEVICE_NAME:
            device = d

    if device is None:
        print("Device not found")
        return

    print("Connecting to", device)

    async with BleakClient(device) as client:
        print("Connected:", client.is_connected)


asyncio.run(main())
