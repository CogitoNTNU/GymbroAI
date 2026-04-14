import asyncio
from bleak import BleakScanner, BleakClient
from dashboard_sync import update_count

DEVICE_NAME = "CogitoIMU"
GEST_UUID = "db6d5260-ae3e-4421-a65c-73ca64cc7d3b"

bicep_curl_counter = 0
shoulder_press_counter = 0
rows_counter = 0
squats_counter = 0
triceps_extension_counter = 0


def gest_callback(sender, data):
    data = data.decode()
    gesture, confidence = data.split("|")
    print(f"Gesture: {gesture}, Confidence: {confidence}")
    if gesture == "bicep_curl" and float(confidence) > 0.8:
        global bicep_curl_counter
        bicep_curl_counter += 1
        update_count("bicep_curl_counter", bicep_curl_counter)
        print(f"Bicep Curl Count: {bicep_curl_counter}")
    elif gesture == "shoulder_press" and float(confidence) > 0.8:
        global shoulder_press_counter
        shoulder_press_counter += 1
        update_count("shoulder_press_counter", shoulder_press_counter)
        print(f"Shoulder Press Count: {shoulder_press_counter}")

    elif gesture == "rows" and float(confidence) > 0.8:
        global rows_counter
        rows_counter += 1
        update_count("rows_counter", rows_counter)
        print(f"Rows Count: {rows_counter}")

    elif gesture == "squat" and float(confidence) > 0.8:
        global squats_counter
        squats_counter += 1
        update_count("squats_counter", squats_counter)
        print(f"Squats Count: {squats_counter}")

    elif gesture == "tricep_extension" and float(confidence) > 0.8:
        global triceps_extension_counter
        triceps_extension_counter += 1
        update_count("triceps_extension_counter", triceps_extension_counter)
        print(f"Triceps Extension Count: {triceps_extension_counter}")

    print(
        f"Current Counts - Bicep Curls: {bicep_curl_counter}, Shoulder Presses: {shoulder_press_counter}, Rows: {rows_counter}, Squats: {squats_counter}, Triceps Extensions: {triceps_extension_counter}"
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
