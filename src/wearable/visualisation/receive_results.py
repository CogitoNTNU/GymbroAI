import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from bleak import BleakScanner, BleakClient
from src.wearable.visualisation.dashboard_sync import update_count

DEVICE_NAME = "CogitoIMU"
GEST_UUID = "db6d5260-ae3e-4421-a65c-73ca64cc7d3b"
COUNTS_FILE = os.path.join(os.path.dirname(__file__), "counts.json")

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
        f"Counts — Bicep: {bicep_curl_counter}, Shoulder: {shoulder_press_counter}, "
        f"Rows: {rows_counter}, Squats: {squats_counter}, Triceps: {triceps_extension_counter}"
    )


# ── HTTP Server ───────────────────────────────────────────────────────────────
class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # silence request logs

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/counts"):
            try:
                data = (
                    json.load(open(COUNTS_FILE)) if os.path.exists(COUNTS_FILE) else {}
                )
            except Exception:
                data = {}
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/reset":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            key = body.get("key", "all")

            global bicep_curl_counter, shoulder_press_counter
            global rows_counter, squats_counter, triceps_extension_counter

            all_keys = [
                "bicep_curl_counter",
                "shoulder_press_counter",
                "rows_counter",
                "squats_counter",
                "triceps_extension_counter",
            ]

            if key == "all":
                keys_to_reset = all_keys
                bicep_curl_counter = shoulder_press_counter = 0
                rows_counter = squats_counter = triceps_extension_counter = 0
            elif key in all_keys:
                keys_to_reset = [key]
                if key == "bicep_curl_counter":
                    bicep_curl_counter = 0
                elif key == "shoulder_press_counter":
                    shoulder_press_counter = 0
                elif key == "rows_counter":
                    rows_counter = 0
                elif key == "squats_counter":
                    squats_counter = 0
                elif key == "triceps_extension_counter":
                    triceps_extension_counter = 0
            else:
                keys_to_reset = []

            for k in keys_to_reset:
                update_count(k, 0)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


def start_http_server():
    server = HTTPServer(("localhost", 8765), DashboardHandler)
    print("Dashboard server running at http://localhost:8765")
    server.serve_forever()


# ── BLE Main ─────────────────────────────────────────────────────────────────
async def main():
    t = threading.Thread(target=start_http_server, daemon=True)
    t.start()

    print("Scanning for devices...")
    device = await BleakScanner.find_device_by_name(DEVICE_NAME, timeout=10.0)
    if device is None:
        print(f"Device '{DEVICE_NAME}' not found.")
        return

    async with BleakClient(device) as client:
        print(f"Connected to {DEVICE_NAME}")
        await client.start_notify(GEST_UUID, gest_callback)
        print("Listening for data... Press Ctrl+C to stop.")
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        await client.stop_notify(GEST_UUID)


asyncio.run(main())
