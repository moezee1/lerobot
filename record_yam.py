"""
Record teleoperation episodes from the YAM robot and push to HuggingFace.

Controls (press in terminal while recording):
  Enter  — save current episode, start next
  R      — discard current episode and re-record
  Escape — stop early, finalize, push to Hub

Usage:
  python record_yam.py ^
      --leader-port COM3 ^
      --repo-id your_hf_username/yam_dataset ^
      --task "Pick up the red cube" ^
      --num-episodes 20 ^
      --camera-index 0 2

  # Resume an existing dataset
  python record_yam.py --resume --repo-id your_hf_username/yam_dataset ...

  # Record without pushing
  python record_yam.py --no-push ...

Run as administrator (needed for USB CAN access).
"""

import argparse
import json
import msvcrt
import os
import sys
import threading
import time
from pathlib import Path

# ── suppress i2rt background thread AssertionErrors on shutdown ──────────────
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

# ── lerobot path ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
MotorChainRobot._check_current_qpos_in_joint_limits = lambda self: None

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.motors_bus import MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── optional: rerun ──────────────────────────────────────────────────────────
try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

# ── optional: pyorbbecsdk ─────────────────────────────────────────────────────
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False

# ── paths ─────────────────────────────────────────────────────────────────────
CALIBRATION_PATH = Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so_leader/leader.json"
YAM_CAL_PATH     = Path.home() / ".cache/huggingface/lerobot/calibration/yam_follower.json"

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--leader-port",   default="COM3")
parser.add_argument("--repo-id",       required=True,  help="HuggingFace repo, e.g. alice/yam_pick")
parser.add_argument("--task",          required=True,  help="Task description logged with every frame")
parser.add_argument("--num-episodes",  type=int, default=10)
parser.add_argument("--episode-time",  type=float, default=30.0, help="Max seconds per episode")
parser.add_argument("--hz",            type=float, default=30.0, help="Recording frequency (Hz)")
parser.add_argument("--scale",         type=float, default=1.0)
parser.add_argument("--camera-index",  type=int, nargs="+", default=None,
                    help="OpenCV camera indices, e.g. --camera-index 0 2")
parser.add_argument("--local-dir",     default=None,
                    help="Local dataset root (default: ~/.cache/huggingface/lerobot/datasets/<repo-id>)")
parser.add_argument("--resume",        action="store_true", help="Resume an existing dataset")
parser.add_argument("--no-push",       action="store_true", help="Skip HuggingFace push at the end")
parser.add_argument("--no-rerun",      action="store_true", help="Disable rerun visualization")
parser.add_argument("--private",       action="store_true", help="Push as private HF repo")
args = parser.parse_args()

LEADER_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
JOINT_TO_YAM  = {"shoulder_pan": 0, "shoulder_lift": 1, "elbow_flex": 2, "wrist_flex": 3, "wrist_roll": 5}
LOOP_DT       = 1.0 / args.hz
USE_RERUN     = HAS_RERUN and not args.no_rerun

OFFSETS_DEG = {"shoulder_pan": 0.0, "shoulder_lift": 0.0,
               "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0}
MAX_SPEED_RAD = np.radians(250.0)


# ── camera classes ────────────────────────────────────────────────────────────
class OrbbecCamera:
    def __init__(self):
        self.pipeline = Pipeline()
        cfg = Config()
        try:
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR)
            try:
                profile = profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            except Exception:
                profile = profiles.get_default_video_stream_profile()
            cfg.enable_stream(profile)
        except Exception as e:
            print(f"  WARNING: Orbbec color stream unavailable: {e}")
        try:
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH)
            cfg.enable_stream(profiles.get_default_video_stream_profile())
        except Exception:
            pass
        self.pipeline.start(cfg)
        self._color = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="orbbec")
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                fs = self.pipeline.wait_for_frames(100)
                if fs is None:
                    continue
                cf = fs.get_color_frame()
                if cf:
                    h, w = cf.get_height(), cf.get_width()
                    data = np.frombuffer(cf.get_data(), dtype=np.uint8)
                    arr = None
                    if len(data) == h * w * 3:
                        arr = data.reshape(h, w, 3).copy()
                    else:
                        import cv2
                        dec = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        if dec is not None:
                            arr = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
                    if arr is not None:
                        with self._lock:
                            self._color = arr
            except Exception:
                pass

    def get_color(self):
        with self._lock:
            return self._color

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVCamera:
    def __init__(self, index: int):
        import cv2
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        ok, _ = self._cap.read()
        if not ok:
            self._cap.release()
            raise RuntimeError(f"Camera index {index} opened but gave no frame")
        self._color = None
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{index}")
        self._thread.start()

    def _loop(self):
        import cv2
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._color = np.ascontiguousarray(rgb)

    def get_color(self):
        with self._lock:
            return self._color

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()


# ── keyboard helpers (non-blocking, Windows) ──────────────────────────────────
def poll_key() -> str | None:
    if msvcrt.kbhit():
        ch = msvcrt.getch()
        try:
            return ch.decode("utf-8").lower()
        except Exception:
            return None
    return None


# ── connect leader ────────────────────────────────────────────────────────────
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

# ── load YAM calibration ──────────────────────────────────────────────────────
if YAM_CAL_PATH.exists():
    yam_cal = json.loads(YAM_CAL_PATH.read_text())
    bad = [j for j, c in yam_cal.items() if abs(c["leader_max"] - c["leader_min"]) < 1.0]
    for j in bad:
        del yam_cal[j]
    if bad:
        print(f"WARNING: skipping zero-range joints: {bad}")
    print(f"YAM calibration loaded for: {list(yam_cal.keys())}")
else:
    yam_cal = None
    print("No YAM calibration found — using fixed scale/offsets.")

# ── connect follower ──────────────────────────────────────────────────────────
print("Connecting to YAM follower...")
follower = get_yam_robot(channel="can0", zero_gravity_mode=False)
n_dofs = follower.num_dofs()
print(f"Follower connected — {n_dofs} DOFs.")
current_cmd = follower.get_joint_pos().copy()

# ── connect cameras ───────────────────────────────────────────────────────────
cameras: list = []
if HAS_ORBBEC:
    try:
        cameras.append(OrbbecCamera())
        print("Orbbec camera connected.")
    except Exception as e:
        print(f"Orbbec unavailable ({e}), using OpenCV only.")

if not cameras or args.camera_index is not None:
    import cv2
    indices = args.camera_index if args.camera_index is not None else range(10)
    for idx in indices:
        try:
            cameras.append(OpenCVCamera(idx))
            print(f"OpenCV camera [{idx}] connected.")
        except Exception as e:
            if args.camera_index is not None:
                print(f"WARNING: camera index {idx}: {e}")

if not cameras:
    print("WARNING: no cameras found — recording without images.")

# ── warm up cameras and discover resolution ───────────────────────────────────
if cameras:
    print("Warming up cameras...", end="", flush=True)
    cam_shapes = {}
    deadline = time.time() + 5.0
    while time.time() < deadline:
        pending = [i for i in range(len(cameras)) if i not in cam_shapes]
        for i in pending:
            frame = cameras[i].get_color()
            if frame is not None:
                h, w = frame.shape[:2]
                cam_shapes[i] = (h, w)
        if len(cam_shapes) == len(cameras):
            break
        time.sleep(0.05)
    print(f" done. Resolutions: { {f'cam_{i}': s for i, s in cam_shapes.items()} }")

# ── build dataset features ────────────────────────────────────────────────────
YAM_JOINT_NAMES = [f"joint_{i+1}" for i in range(n_dofs - 1)] + ["gripper"]
LEADER_NAMES    = [*LEADER_JOINTS, "gripper"]

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (n_dofs,),
        "names": YAM_JOINT_NAMES,
    },
    "observation.leader_state": {
        "dtype": "float32",
        "shape": (len(LEADER_NAMES),),
        "names": LEADER_NAMES,
    },
    "action": {
        "dtype": "float32",
        "shape": (n_dofs,),
        "names": YAM_JOINT_NAMES,
    },
}
for i, (h, w) in cam_shapes.items():
    features[f"observation.images.cam_{i}"] = {
        "dtype": "video",
        "shape": (3, h, w),
        "names": ["channels", "height", "width"],
    }

# ── create / resume dataset ───────────────────────────────────────────────────
local_dir = Path(args.local_dir) if args.local_dir else None

if args.resume:
    print(f"Resuming dataset {args.repo_id}...")
    dataset = LeRobotDataset.resume(
        repo_id=args.repo_id,
        root=local_dir,
    )
else:
    print(f"Creating dataset {args.repo_id}...")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(args.hz),
        features=features,
        robot_type="yam",
        use_videos=True,
        image_writer_threads=max(1, len(cameras) * 2),
        root=local_dir,
    )

# ── rerun ─────────────────────────────────────────────────────────────────────
if USE_RERUN:
    rr.init("record_yam", spawn=True)
    time.sleep(1.0)
    print("Rerun viewer launched.")

# ── recording loop ────────────────────────────────────────────────────────────
print(f"""
Dataset : {args.repo_id}
Task    : {args.task}
Episodes: {args.num_episodes}
Duration: {args.episode_time}s  @ {args.hz} Hz
Cameras : {len(cameras)}

Controls:
  Enter  — save episode, start next
  R      — discard episode, re-record
  Escape — stop now and push to Hub
""")

total_saved = 0

try:
    while total_saved < args.num_episodes:
        ep_num = total_saved + 1
        print(f"\n── Episode {ep_num}/{args.num_episodes} ──  (Enter=save  R=redo  Esc=quit)")

        ep_start   = time.time()
        frame_idx  = 0
        save_ep    = False
        redo_ep    = False
        stop_all   = False
        latest_frames = {i: None for i in range(len(cameras))}

        while True:
            t0 = time.time()
            elapsed = t0 - ep_start

            # Keyboard
            key = poll_key()
            if key == "\r" or key == "\n":      # Enter
                save_ep = True
                break
            elif key == "r":
                redo_ep = True
                break
            elif key == "\x1b":                 # Escape
                save_ep = True
                stop_all = True
                break

            if elapsed >= args.episode_time:
                print(f"\n  Time limit reached — saving episode.")
                save_ep = True
                break

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

            cmd = current_cmd.copy()
            max_step = MAX_SPEED_RAD * LOOP_DT
            delta = np.clip(target[:-1] - current_cmd[:-1], -max_step, max_step)
            cmd[:-1] = current_cmd[:-1] + delta
            cmd[-1]  = target[-1]

            follower.command_joint_pos(cmd)
            current_cmd = cmd

            # Grab camera frames
            for i, cam in enumerate(cameras):
                f = cam.get_color()
                if f is not None:
                    latest_frames[i] = f

            # Build dataset frame
            frame = {
                "task": args.task,
                "observation.state": current_cmd.astype(np.float32),
                "action": cmd.astype(np.float32),
                "observation.leader_state": np.array(
                    [float(obs[j]) for j in LEADER_NAMES], dtype=np.float32
                ),
            }
            for i in range(len(cameras)):
                key_name = f"observation.images.cam_{i}"
                if key_name in features and latest_frames[i] is not None:
                    frame[key_name] = latest_frames[i]

            # Only add frame if all image features have data
            missing = [k for k in features if k.startswith("observation.images") and k not in frame]
            if not missing:
                dataset.add_frame(frame)
                frame_idx += 1

            # Rerun logging
            if USE_RERUN:
                rr.set_time("wall_clock", timestamp=t0)
                for i, img in latest_frames.items():
                    if img is not None:
                        rr.log(f"camera/{i}/color", rr.Image(np.ascontiguousarray(img)))
                for joint in LEADER_JOINTS:
                    rr.log(f"joints/leader/{joint}", rr.Scalars(float(obs[joint])))
                rr.log("joints/leader/gripper", rr.Scalars(float(obs["gripper"])))
                for joint in LEADER_JOINTS:
                    rr.log(f"joints/follower/{joint}", rr.Scalars(float(np.degrees(cmd[JOINT_TO_YAM[joint]]))))
                rr.log("joints/follower/gripper", rr.Scalars(float(cmd[-1])))

            # Status
            leader_str   = " ".join(f"{float(obs[j]):+6.1f}" for j in LEADER_JOINTS)
            follower_str = " ".join(f"{np.degrees(v):+6.1f}" for v in cmd[:5])
            print(f"\r  t={elapsed:5.1f}s  fr={frame_idx:4d}  "
                  f"L:[{leader_str}]  F:[{follower_str}]", end="", flush=True)

            dt = time.time() - t0
            if LOOP_DT - dt > 0:
                time.sleep(LOOP_DT - dt)

        print()  # newline after status line

        if redo_ep:
            print(f"  Discarding episode {ep_num} ({frame_idx} frames).")
            dataset.clear_episode_buffer()
            continue

        if save_ep and frame_idx > 0:
            print(f"  Saving episode {ep_num} ({frame_idx} frames)...")
            dataset.save_episode()
            total_saved += 1
            print(f"  Saved. Total episodes: {total_saved}")
        elif frame_idx == 0:
            print("  No frames recorded — skipping save.")
            dataset.clear_episode_buffer()

        if stop_all:
            break

except KeyboardInterrupt:
    print("\nInterrupted — finalizing.")
    if dataset.episode_buffer.get("size", 0) > 0:
        print("  Discarding partial episode.")
        dataset.clear_episode_buffer()

finally:
    for cam in cameras:
        cam.stop()
    leader_bus.disconnect()
    follower.close()

    print(f"\nFinalizing dataset ({total_saved} episodes)...")
    dataset.finalize()

    if not args.no_push and total_saved > 0:
        print(f"Pushing to HuggingFace: {args.repo_id} ...")
        dataset.push_to_hub(private=args.private)
        print("Done! Dataset available at:")
        print(f"  https://huggingface.co/datasets/{args.repo_id}")
    elif total_saved == 0:
        print("Nothing to push (0 episodes recorded).")
    else:
        print("Skipping push (--no-push).")

    print("All done.")
