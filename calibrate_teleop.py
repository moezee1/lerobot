"""
LeRobot-style min/max calibration for SO-100 leader -> YAM follower.

For each joint you'll be prompted to move both arms to the MIN position,
press Enter, then to the MAX position, press Enter. The script records the
linear mapping from leader degrees to YAM radians and saves it to:
  ~/.cache/huggingface/lerobot/calibration/yam_follower.json

Usage:
  python calibrate_teleop.py --leader-port COM3

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

logging.disable(logging.CRITICAL)  # silence i2rt / CAN noise

# suppress background thread errors from i2rt on shutdown
_orig_excepthook = threading.excepthook
def _quiet_thread_excepthook(args):
    if args.exc_type is AssertionError:
        return
    _orig_excepthook(args)
threading.excepthook = _quiet_thread_excepthook

# ── libusb / CAN setup ──────────────────────────────────────────────────────
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

# joint name -> YAM joint index
JOINT_MAP = {
    "shoulder_pan":  0,
    "shoulder_lift": 1,
    "elbow_flex":    2,
    "wrist_flex":    3,
    "wrist_roll":    5,
}

parser = argparse.ArgumentParser()
parser.add_argument("--leader-port", default="COM3")
args = parser.parse_args()

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
all_motors = [*JOINT_MAP.keys(), "gripper"]
leader_bus.disable_torque()  # make leader backdrivable during calibration
print("Leader connected.")

print("Connecting to YAM follower arm (gravity-comp)...")
follower = get_yam_robot(channel="can0", zero_gravity_mode=True)
print("Follower connected.\n")


def read_both():
    obs = {j: float(leader_bus.read("Present_Position", j))
           for j in [*JOINT_MAP.keys(), "gripper"]}
    yam = follower.get_joint_pos()
    return obs, yam


def wait_for_enter(leader_key, yam_idx, is_gripper=False):
    """Live-display only the relevant joint until Enter. Returns (obs, yam)."""
    ready = threading.Event()
    threading.Thread(target=lambda: (input(), ready.set()), daemon=True).start()
    snapshot = None
    while not ready.is_set():
        obs, yam = read_both()
        if is_gripper:
            line = f"  leader: {obs['gripper']:5.1f}  |  yam: {yam[-1]:.3f}"
        else:
            line = f"  leader: {obs[leader_key]:+7.2f}°  |  yam: {np.degrees(yam[yam_idx]):+7.2f}°"
        print(line + "   (Enter to record)", end="\r", flush=True)
        snapshot = (obs, yam)
        time.sleep(0.05)
    print()
    return snapshot


cal = {}
steps = list(JOINT_MAP.items()) + [("gripper", -1)]
total = len(steps)

print(f"\nReady — {total} joints to calibrate. Move both arms together for each step.\n")

try:
    for step_num, (joint, yam_idx) in enumerate(steps, 1):
        is_gripper = (joint == "gripper")
        label = f"YAM j{yam_idx+1}" if not is_gripper else "gripper"
        print(f"[{step_num}/{total}]  {joint}  ({label})")

        print("       → move to MINIMUM position")
        obs_min, yam_min_snap = wait_for_enter(joint, yam_idx, is_gripper)

        print("       → move to MAXIMUM position")
        obs_max, yam_max_snap = wait_for_enter(joint, yam_idx, is_gripper)

        if is_gripper:
            lmin, lmax = float(obs_min["gripper"]), float(obs_max["gripper"])
            ymin, ymax = float(yam_min_snap[-1]), float(yam_max_snap[-1])
            print(f"       saved: leader {lmin:.1f}→{lmax:.1f}  |  yam {ymin:.3f}→{ymax:.3f}\n")
            cal["gripper"] = {"leader_min": lmin, "leader_max": lmax, "yam_min": ymin, "yam_max": ymax}
        else:
            lmin, lmax = obs_min[joint], obs_max[joint]
            ymin, ymax = float(yam_min_snap[yam_idx]), float(yam_max_snap[yam_idx])
            if abs(np.degrees(ymax) - np.degrees(ymin)) < 1.0:
                print(f"       WARNING: YAM range too small ({np.degrees(ymax - ymin):.2f}°) — skipping\n")
                continue
            print(f"       saved: leader {lmin:+.1f}°→{lmax:+.1f}°  |  yam {np.degrees(ymin):+.1f}°→{np.degrees(ymax):+.1f}°\n")
            cal[joint] = {"leader_min": lmin, "leader_max": lmax, "yam_min": ymin, "yam_max": ymax}

except KeyboardInterrupt:
    print("\nAborted — nothing saved.")
    leader_bus.enable_torque()
    leader_bus.disconnect()
    follower.close()
    sys.exit(0)

# ── save ─────────────────────────────────────────────────────────────────────
YAM_CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
YAM_CAL_PATH.write_text(json.dumps(cal, indent=2))
print(f"\nCalibration saved to {YAM_CAL_PATH}")

leader_bus.enable_torque()
leader_bus.disconnect()
follower.close()
print("Done.")
