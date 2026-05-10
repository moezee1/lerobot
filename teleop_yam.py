"""
Teleoperation: Feetech SO-100/101 leader arm -> i2RT YAM follower arm.

Usage:
  python teleop_yam.py --leader-port COM3

  Requires: pip install pyorbbecsdk rerun-sdk
  Orbbec Gemini 2 must have its SDK drivers installed (OrbbecSDK_v2).

Joint mapping (SO leader -> YAM follower):
  shoulder_pan   -> joint 1
  shoulder_lift  -> joint 2
  elbow_flex     -> joint 3
  wrist_flex     -> joint 4
  wrist_roll     -> joint 5
  (joint 6 held at 0)
  gripper (0-100) -> gripper (0-1)

Run as administrator (needed for USB CAN access).
"""

import argparse
import os
import sys
import threading
import time

# suppress i2rt background thread AssertionErrors on shutdown
_orig_thread_excepthook = threading.excepthook
def _quiet_thread_excepthook(args):
    if args.exc_type is AssertionError:
        return
    _orig_thread_excepthook(args)
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

# ── imports ─────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(__file__.replace("teleop_yam.py", "src")))

import numpy as np
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
MotorChainRobot._check_current_qpos_in_joint_limits = lambda self: None

import json
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus

# ── optional: rerun ─────────────────────────────────────────────────────────
try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False
    print("WARNING: rerun-sdk not found. Run: pip install rerun-sdk")

# ── optional: Orbbec Gemini 2 ───────────────────────────────────────────────
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False
    print("WARNING: pyorbbecsdk not found. Run: pip install pyorbbecsdk")

CALIBRATION_PATH = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader.json"
YAM_CAL_PATH     = Path.home() / ".cache/huggingface/lerobot/calibration/yam_follower.json"

# ── args ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--leader-port", default="COM3", help="Serial port of the SO leader arm")
parser.add_argument("--hz", type=float, default=50.0, help="Control loop frequency (Hz)")
parser.add_argument("--scale", type=float, default=1.0,
                    help="Scale factor applied to leader joint angles (default 1.0)")
parser.add_argument("--no-camera", action="store_true", help="Disable camera")
parser.add_argument("--no-rerun", action="store_true", help="Disable rerun visualization")
parser.add_argument("--camera-index", type=int, nargs="+", default=None,
                    help="One or more OpenCV camera indices, e.g. --camera-index 0 2")
args = parser.parse_args()

LEADER_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_TO_YAM = {"shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2, "wrist_flex": 3, "wrist_roll": 5}
LOOP_DT = 1.0 / args.hz

USE_CAMERA = not args.no_camera
USE_RERUN  = HAS_RERUN and not args.no_rerun


# ── Orbbec Gemini 2 camera ──────────────────────────────────────────────────
class OrbbecCamera:
    """Captures color + depth from Orbbec Gemini 2 in a background thread."""

    def __init__(self):
        self.pipeline = Pipeline()
        cfg = Config()

        # Color stream — prefer explicit RGB; fall back to default
        try:
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR)
            try:
                profile = profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            except Exception:
                profile = profiles.get_default_video_stream_profile()
            cfg.enable_stream(profile)
            print(f"  Camera color: {profile.get_width()}x{profile.get_height()} "
                  f"@ {profile.get_fps()} fps  fmt={profile.get_format()}")
        except Exception as e:
            print(f"  WARNING: color stream unavailable: {e}")

        # Depth stream
        try:
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH)
            profile = profiles.get_default_video_stream_profile()
            cfg.enable_stream(profile)
            print(f"  Camera depth: {profile.get_width()}x{profile.get_height()} "
                  f"@ {profile.get_fps()} fps")
        except Exception as e:
            print(f"  WARNING: depth stream unavailable: {e}")

        self.pipeline.start(cfg)

        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="orbbec-capture")
        self._thread.start()

    def _decode_color(self, frame) -> np.ndarray | None:
        h, w = frame.get_height(), frame.get_width()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        n = len(data)
        if n == h * w * 3:
            # RGB or BGR — return as-is (requested RGB above)
            return data.reshape(h, w, 3).copy()
        if n == h * w * 2:
            # YUYV
            try:
                import cv2
                return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUYV)
            except Exception:
                return None
        # MJPEG or other compressed format
        try:
            import cv2
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        return None

    def _loop(self):
        while self._running:
            try:
                frameset = self.pipeline.wait_for_frames(100)
                if frameset is None:
                    continue

                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()

                color_arr = self._decode_color(color_frame) if color_frame else None

                depth_arr = None
                if depth_frame:
                    h, w = depth_frame.get_height(), depth_frame.get_width()
                    data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                    if len(data) == h * w:
                        depth_arr = data.reshape(h, w).copy()

                with self._lock:
                    if color_arr is not None:
                        self._color = color_arr
                    if depth_arr is not None:
                        self._depth = depth_arr
            except Exception:
                pass

    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            return self._color, self._depth

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ── OpenCV fallback camera (color only) ─────────────────────────────────────
class OpenCVCamera:
    """Background thread that grabs color frames via OpenCV (UVC fallback)."""

    def __init__(self, index: int):
        import cv2
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  OpenCV camera [{index}]: {w}x{h}")
        self._color: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="opencv-capture")
        self._thread.start()

    def _loop(self):
        import cv2
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._color = np.ascontiguousarray(rgb)

    def get_frames(self) -> tuple[np.ndarray | None, None]:
        with self._lock:
            return self._color, None

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()


# ── connect leader ───────────────────────────────────────────────────────────
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
_cal_raw = json.loads(CALIBRATION_PATH.read_text())
leader_bus.calibration = {k: MotorCalibration(**v) for k, v in _cal_raw.items()}
print("Leader connected.")

# ── load YAM calibration (optional) ─────────────────────────────────────────
if YAM_CAL_PATH.exists():
    yam_cal = json.loads(YAM_CAL_PATH.read_text())
    bad = [j for j, c in yam_cal.items() if abs(c["leader_max"] - c["leader_min"]) < 1.0]
    for j in bad:
        del yam_cal[j]
    if bad:
        print(f"WARNING: skipping zero-range joints (re-run calibrate_joint.py): {bad}")
    print(f"YAM calibration loaded for: {list(yam_cal.keys())}")
else:
    yam_cal = None
    print("No YAM calibration found — using fixed scale/offsets. Run calibrate_teleop.py to calibrate.")

# ── connect follower ─────────────────────────────────────────────────────────
print("Connecting to YAM follower arm...")
follower = get_yam_robot(channel="can0", zero_gravity_mode=False)
n_dofs = follower.num_dofs()
print(f"Follower connected — {n_dofs} DOFs.")

current_cmd = follower.get_joint_pos().copy()

# ── calibration offsets ──────────────────────────────────────────────────────
OFFSETS_DEG = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
}

MAX_SPEED_RAD = np.radians(250.0)

# ── rerun init ───────────────────────────────────────────────────────────────
if USE_RERUN:
    rr.init("teleop_yam", spawn=True)
    time.sleep(1.0)  # let the viewer connect before we start logging
    print("Rerun viewer launched.")
elif not HAS_RERUN:
    pass  # warning already printed
else:
    print("Rerun disabled (--no-rerun).")

# ── camera init ──────────────────────────────────────────────────────────────
cameras: list = []
if USE_CAMERA:
    import cv2
    if HAS_ORBBEC:
        print("Connecting to Orbbec Gemini 2 (pyorbbecsdk)...")
        try:
            cameras.append(OrbbecCamera())
            print("Orbbec camera connected.")
        except Exception as e:
            print(f"WARNING: pyorbbecsdk failed ({e}), using OpenCV only.")

    if not cameras or args.camera_index is not None:
        # Use explicit indices or auto-detect
        indices = args.camera_index if args.camera_index is not None else range(10)
        for idx in indices:
            try:
                cam = OpenCVCamera(idx)
                cameras.append(cam)
                print(f"Camera connected via OpenCV (index {idx}).")
            except Exception:
                if args.camera_index is not None:
                    print(f"WARNING: could not open camera index {idx}.")

    if not cameras:
        print("WARNING: no cameras found. Use --camera-index N [N ...] to specify.")
else:
    print("Camera disabled (--no-camera).")

print(f"\nTeleop running at {args.hz} Hz — Ctrl+C to stop.\n")
print("  Leader joints (deg) -> Follower joints (rad)")

try:
    while True:
        t0 = time.time()

        # Read leader
        obs = {j: leader_bus.read("Present_Position", j) for j in [*LEADER_JOINTS, "gripper"]}

        # Build follower command
        target = current_cmd.copy()
        for joint in LEADER_JOINTS:
            yi = JOINT_TO_YAM[joint]
            if yam_cal and joint in yam_cal:
                c = yam_cal[joint]
                t = np.clip((float(obs[joint]) - c["leader_min"]) / (c["leader_max"] - c["leader_min"]), 0.0, 1.0)
                target[yi] = c["yam_min"] + t * (c["yam_max"] - c["yam_min"])
            else:
                target[yi] = np.radians(-float(obs[joint]) * args.scale + OFFSETS_DEG[joint])
        if yam_cal and "gripper" in yam_cal:
            c = yam_cal["gripper"]
            t = np.clip((float(obs["gripper"]) - c["leader_min"]) / (c["leader_max"] - c["leader_min"]), 0.0, 1.0)
            target[-1] = c["yam_min"] + t * (c["yam_max"] - c["yam_min"])
        else:
            target[-1] = float(obs["gripper"]) / 100.0

        # Rate-limit arm joints
        cmd = current_cmd.copy()
        max_step = MAX_SPEED_RAD * LOOP_DT
        delta = np.clip(target[:-1] - current_cmd[:-1], -max_step, max_step)
        cmd[:-1] = current_cmd[:-1] + delta
        cmd[-1] = target[-1]

        follower.command_joint_pos(cmd)
        current_cmd = cmd

        # ── rerun logging ────────────────────────────────────────────────────
        if USE_RERUN:
            rr.set_time("wall_clock", timestamp=t0)

            # Camera frames
            for i, cam in enumerate(cameras):
                color, depth = cam.get_frames()
                if color is not None:
                    rr.log(f"camera/{i}/color", rr.Image(np.ascontiguousarray(color)))
                if depth is not None:
                    rr.log(f"camera/{i}/depth", rr.DepthImage(depth, meter=1000.0))

            # Leader joint angles (degrees)
            for joint in LEADER_JOINTS:
                rr.log(f"joints/leader/{joint}", rr.Scalars(float(obs[joint])))
            rr.log("joints/leader/gripper", rr.Scalars(float(obs["gripper"])))

            # Follower joint angles (degrees)
            for joint in LEADER_JOINTS:
                yi = JOINT_TO_YAM[joint]
                rr.log(f"joints/follower/{joint}", rr.Scalars(float(np.degrees(cmd[yi]))))
            rr.log("joints/follower/gripper", rr.Scalars(float(cmd[-1])))

        # Status line
        leader_str = " ".join(f"{float(obs[j]):+6.1f}" for j in LEADER_JOINTS)
        follower_str = " ".join(f"{np.degrees(v):+6.1f}" for v in cmd[:5])
        print(f"\r  L:[{leader_str}]  F:[{follower_str}]", end="", flush=True)

        elapsed = time.time() - t0
        sleep_t = LOOP_DT - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

except KeyboardInterrupt:
    print("\nStopping.")

finally:
    for cam in cameras:
        cam.stop()
    leader_bus.disconnect()
    follower.close()
    print("Done.")
