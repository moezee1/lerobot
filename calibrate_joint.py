"""
Calibrate a single joint and merge it into the existing yam_follower.json.

Usage:
  python calibrate_joint.py --joint wrist_roll --leader-port COM3

Run as administrator.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

logging.disable(logging.CRITICAL)

import threading as _threading
_orig_excepthook = _threading.excepthook
def _quiet(args):
    if args.exc_type is AssertionError:
        return
    _orig_excepthook(args)
_threading.excepthook = _quiet

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

from i2rt.motor_drivers import can_interface as _ci
def _patched_send(self, id, motor_id, data, max_retry=10, expected_id=None):
    msg = can.Message(arbitration_id=id, data=data, is_extended_id=False)
    if expected_id is None:
        expected_id = self.receive_mode.get_receive_id(motor_id)
    for _ in range(max_retry):
        try:
            self.bus.send(msg)
            deadline = time.time() + 0.15
            while time.time() < deadline:
                r = self.bus.recv(timeout=0.005)
                if r and r.arbitration_id == expected_id:
                    return r
        except Exception:
            pass
        time.sleep(0.005)
    raise AssertionError(f"no response from motor {motor_id}")
_ci.CanInterface._send_message_get_response = _patched_send

import numpy as np
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
MotorChainRobot._check_current_qpos_in_joint_limits = lambda self: None

sys.path.insert(0, str(Path(__file__).parent / "src"))
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus

LEADER_CAL_PATH = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader.json"
YAM_CAL_PATH    = Path.home() / ".cache/huggingface/lerobot/calibration/yam_follower.json"

JOINT_MAP = {
    "shoulder_pan":  0,
    "shoulder_lift": 1,
    "elbow_flex":    2,
    "wrist_flex":    3,
    "wrist_roll":    5,
    "gripper":      -1,
}

parser = argparse.ArgumentParser()
parser.add_argument("--joint", required=True, choices=list(JOINT_MAP.keys()), help="Joint to calibrate")
parser.add_argument("--leader-port", default="COM3")
args = parser.parse_args()

joint = args.joint
yam_idx = JOINT_MAP[joint]
is_gripper = (joint == "gripper")

# ── connect ──────────────────────────────────────────────────────────────────
print(f"Connecting to SO leader on {args.leader_port}...")
leader_bus = FeetechMotorsBus(
    port=args.leader_port,
    motors={
        "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
)
leader_bus.connect()
_cal_raw = json.loads(LEADER_CAL_PATH.read_text())
leader_bus.calibration = {k: MotorCalibration(**v) for k, v in _cal_raw.items()}
leader_bus.disable_torque()
print("Leader connected.")

print("Connecting to YAM follower arm (gravity-comp)...")
follower = get_yam_robot(channel="can0", zero_gravity_mode=True)
print(f"Follower connected.\n")


def read_both():
    leader_val = float(leader_bus.read("Present_Position", joint))
    yam = follower.get_joint_pos()
    yam_val = yam[-1] if is_gripper else yam[yam_idx]
    return leader_val, float(yam_val)


def wait_for_enter(prompt):
    ready = threading.Event()
    threading.Thread(target=lambda: (input(), ready.set()), daemon=True).start()
    print(prompt)
    snapshot = None
    while not ready.is_set():
        lv, yv = read_both()
        if is_gripper:
            line = f"  leader: {lv:5.1f}  |  yam: {yv:.3f}"
        else:
            line = f"  leader: {lv:+7.2f}°  |  yam: {np.degrees(yv):+7.2f}°"
        print(line + "   (Enter to record)", end="\r", flush=True)
        snapshot = (lv, yv)
        time.sleep(0.05)
    print()
    return snapshot


try:
    label = f"YAM j{yam_idx+1}" if not is_gripper else "gripper"
    print(f"Calibrating: {joint}  ({label})\n")

    lv_min, yv_min = wait_for_enter("  → move to MINIMUM position")
    print(f"  MIN: leader={lv_min:+.2f}  yam={np.degrees(yv_min):+.2f}°" if not is_gripper
          else f"  MIN: leader={lv_min:.1f}  yam={yv_min:.3f}")

    lv_max, yv_max = wait_for_enter("  → move to MAXIMUM position")
    print(f"  MAX: leader={lv_max:+.2f}  yam={np.degrees(yv_max):+.2f}°" if not is_gripper
          else f"  MAX: leader={lv_max:.1f}  yam={yv_max:.3f}")

except KeyboardInterrupt:
    print("\nAborted — nothing saved.")
    leader_bus.enable_torque()
    leader_bus.disconnect()
    follower.close()
    sys.exit(0)

# ── merge into existing cal file ─────────────────────────────────────────────
cal = json.loads(YAM_CAL_PATH.read_text()) if YAM_CAL_PATH.exists() else {}
cal[joint] = {
    "leader_min": lv_min,
    "leader_max": lv_max,
    "yam_min":    yv_min,
    "yam_max":    yv_max,
}
YAM_CAL_PATH.write_text(json.dumps(cal, indent=2))
print(f"\nSaved '{joint}' to {YAM_CAL_PATH}")

leader_bus.enable_torque()
leader_bus.disconnect()
follower.close()
print("Done.")
