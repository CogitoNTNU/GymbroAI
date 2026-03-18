import asyncio
from bleak import BleakClient

address = "42:ED:1F:D4:F4:03"
MODEL_NBR_UUID = "12345678-1234-1234-1234-1234567890ac"


async def main(address):
    async with BleakClient(address) as client:
        model_number = await client.read_gatt_char(MODEL_NBR_UUID)
        print(f"Model Number: {model_number.decode()}")


asyncio.run(main(address))
