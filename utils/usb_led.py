import os
import sys

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed. Arduino LED control will be disabled. Run: pip install pyserial")

class USBLEDController:
    """
    Controls a USB LED light based on animal detection events.
    Communicates with Arduino via Serial.
    """
    def __init__(self):
        self.is_on = False
        self.serial_conn = None
        self._connect_arduino()

    def _connect_arduino(self):
        if not SERIAL_AVAILABLE:
            return

        # Attempt to auto-detect Arduino port
        ports = list(serial.tools.list_ports.comports())
        target_port = None
        
        # Look for typical Arduino identifiers in description or name
        for p in ports:
            desc = p.description.lower()
            name = p.device.lower()
            if "arduino" in desc or "ch340" in desc or "usbmodem" in name or "ttyacm" in name or "ttyusb" in name:
                target_port = p.device
                break
        
        # Fallback to the first available if not explicitly found, or leave as None
        if target_port is None and len(ports) > 0:
            target_port = ports[0].device

        if target_port:
            try:
                self.serial_conn = serial.Serial(target_port, 9600, timeout=1)
                print(f"[INFO] Connected to Arduino LED controller on {target_port}")
            except Exception as e:
                print(f"[ERROR] Failed to connect to Arduino on {target_port}: {e}")
        else:
            print("[WARN] No Arduino port detected for LED controller.")

    def turn_on(self):
        if not self.is_on:
            print("[INFO] Turning Arduino LEDs ON.")
            self._execute_command("ON")
            self.is_on = True

    def turn_off(self):
        if self.is_on:
            print("[INFO] Turning Arduino LEDs OFF.")
            self._execute_command("OFF")
            self.is_on = False

    def _execute_command(self, state: str):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                if state == "ON":
                    self.serial_conn.write(b'1')
                else:
                    self.serial_conn.write(b'0')
            except Exception as e:
                print(f"[ERROR] Serial write failed: {e}")

# Global instance for easy import
led_controller = USBLEDController()
