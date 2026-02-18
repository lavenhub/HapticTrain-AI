import serial
import time

class HapticController:
    def __init__(self, port=None, baudrate=115200):
        self.port = port
        self.ser = None
        
        if port:
            try:
                self.ser = serial.Serial(port, baudrate, timeout=1)
                print(f"Connected to ESP32 on {port}")
            except:
                print("Hardware not found. Entering Simulation Mode...")

    def trigger_vibration(self):
        if self.ser:
            # Send character '1' to ESP32 to buzz
            self.ser.write(b'1')
            return "SIGNAL SENT TO HARDWARE"
        else:
            return "SIMULATED VIBRATION (BUZZ!)"