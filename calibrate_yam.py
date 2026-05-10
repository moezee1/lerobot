"""
YAM arm zero-position calibration (Windows gs_usb).

Procedure:
  1. Power off the arm, move it by hand to the desired zero/home position.
  2. Power the arm back on.
  3. Run this script — it saves the current position as zero for each motor.
  4. Power-cycle the arm after calibration.

Usage:
  python calibrate_yam.py              # calibrate all 7 motors
  python calibrate_yam.py --motor 3   # calibrate only motor 3
"""

import argparse
import os
import time

import libusb
os.environ["PATH"] = os.path.dirname(libusb.dll._name) + os.pathsep + os.environ.get("PATH", "")

import can
import can.interface

_orig_bus = can.interface.Bus
def _patched_bus(*args, **kwargs):
    if kwargs.get("bustype") == "socketcan" or kwargs.get("interface") == "socketcan":
        kwargs.pop("bustype", None)
        kwargs.pop("interface", None)
        kwargs.pop("channel", None)
        kwargs["interface"] = "gs_usb"
        kwargs["channel"] = 0
        kwargs["index"] = 0
        kwargs.setdefault("bitrate", 1000000)
    return _orig_bus(*args, **kwargs)

can.interface.Bus = _patched_bus
can.Bus = _patched_bus

# Patch CAN send/receive for reliable gs_usb communication on Windows
from i2rt.motor_drivers import can_interface as _ci

def _patched_send(self, id, motor_id, data, max_retry=10, expected_id=None):
    message = can.Message(arbitration_id=id, data=data, is_extended_id=False)
    if expected_id is None:
        expected_id = self.receive_mode.get_receive_id(motor_id)
    for _ in range(max_retry):
        try:
            self.bus.send(message)
            deadline = time.time() + 0.15
            while time.time() < deadline:
                response = self.bus.recv(timeout=0.005)
                if response and response.arbitration_id == expected_id:
                    return response
        except (can.CanError, AssertionError):
            pass
        time.sleep(0.005)
    raise AssertionError(f"no response from motor {motor_id}")

_ci.CanInterface._send_message_get_response = _patched_send

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

parser = argparse.ArgumentParser()
parser.add_argument("--motor", type=int, default=-1, help="Motor ID to calibrate (-1 = all)")
args = parser.parse_args()

motor_ids = [args.motor] if args.motor > 0 else [1, 2, 3, 4, 5, 6, 7]

print(f"Calibrating motors: {motor_ids}")
print("Make sure the arm is powered on and in the desired zero/home position.\n")

iface = DMSingleMotorCanInterface(
    channel="can0", bustype="socketcan", control_mode=ControlMode.MIT
)

try:
    for motor_id in motor_ids:
        print(f"  Motor {motor_id}: ", end="", flush=True)
        try:
            iface.motor_on(motor_id, MotorType.DM4310)
            time.sleep(0.05)
            pos_before = iface.set_control(motor_id, MotorType.DM4310, 0, 0, 0, 0, 0).position
            print(f"pos = {pos_before:+.4f} rad  ->  saving zero ...", end="", flush=True)
            iface.save_zero_position(motor_id)
            time.sleep(0.1)
            pos_after = iface.set_control(motor_id, MotorType.DM4310, 0, 0, 0, 0, 0).position
            print(f"  done  (after = {pos_after:+.4f} rad)")
        except Exception as e:
            print(f"  FAILED: {e}")
finally:
    iface.close()

print("\nCalibration complete. Power-cycle the arm now.")
