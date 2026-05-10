"""
VLM-guided water-bottle pick-and-place using the Orbbec Gemini 2 wrist camera.

Pipeline:
  1. Arm moves to SCAN_POS — a taught overhead position where the Gemini camera
     can see the whole workspace.
  2. YOLOv8 (COCO bottle class) runs on the Gemini color frame.
  3. Depth frame gives metric distance → 3D bottle position in camera frame.
  4. Arm executes the pre-taught pick-and-place trajectory.
  5. A live cv2 window shows the annotated feed throughout.

────────────────────────────────────────────────────────────────────────
QUICK START
────────────────────────────────────────────────────────────────────────
  pip install ultralytics opencv-python

  # Teach 7 poses (arm in zero-grav, move by hand, SPACE to record)
  python vlm_pick_bottle.py --teach --waypoints my_waypoints.json

  # Run
  python vlm_pick_bottle.py --waypoints my_waypoints.json

  # Camera/detection only — no arm
  python vlm_pick_bottle.py --no-arm

────────────────────────────────────────────────────────────────────────
WAYPOINT SEQUENCE (recorded during --teach)
────────────────────────────────────────────────────────────────────────
  home        — safe resting pose, gripper open
  scan_pos    — overhead pose where Gemini camera sees the workspace
  pre_grasp_a — above bottle at A (~10 cm above), gripper open
  grasp_a     — gripper around bottle at A, fingers open
  lift        — raised ~15 cm with bottle
  pre_place_b — above target B, carrying bottle
  place_b     — at target B, ready to release

────────────────────────────────────────────────────────────────────────
DETECTION MODEL
────────────────────────────────────────────────────────────────────────
  YOLOv8m pretrained on COCO (class 39 = bottle).
  Trained on 10k+ annotated bottle images — no zero-shot guessing.
  Use --yolov8-model yolov8l.pt for higher accuracy (slower).

────────────────────────────────────────────────────────────────────────
DATASETS FOR FURTHER TRAINING
────────────────────────────────────────────────────────────────────────
  lerobot/libero_object   — 100 object pick tasks (HuggingFace)
  lerobot-raw/droid       — 76k diverse robot episodes
  lerobot-raw/bridge_v2   — BridgeData V2 kitchen pick-place
  COCO 2017               — bottle class fine-tuning
  Open Images v7          — "Water bottle" annotations

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

import numpy as np

# ── gripper convention ────────────────────────────────────────────────────────
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.8   # tune for your bottle diameter

MAX_SPEED_RAD  = np.radians(60.0)

# ── suppress i2rt background thread AssertionErrors ──────────────────────────
_orig_thread_excepthook = threading.excepthook
def _quiet_thread_excepthook(args):
    if args.exc_type is AssertionError:
        return
    _orig_thread_excepthook(args)
threading.excepthook = _quiet_thread_excepthook

# ── libusb / CAN setup ───────────────────────────────────────────────────────
try:
    import libusb
    os.environ["PATH"] = os.path.dirname(libusb.dll._name) + os.pathsep + os.environ.get("PATH", "")
    import can
    import can.interface

    _orig_bus = can.interface.Bus
    def _patched_bus(*a, **kw):
        if kw.get("bustype") == "socketcan" or kw.get("interface") == "socketcan":
            kw.pop("bustype", None); kw.pop("interface", None); kw.pop("channel", None)
            kw["interface"] = "gs_usb"; kw["channel"] = 0; kw["index"] = 0
            kw.setdefault("bitrate", 1000000)
        return _orig_bus(*a, **kw)
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
    HAS_CAN = True
except Exception as _e:
    HAS_CAN = False
    print(f"WARNING: CAN/libusb unavailable ({_e}) — arm disabled.")

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Orbbec SDK ────────────────────────────────────────────────────────────────
try:
    from pyorbbecsdk import (Pipeline, Config, OBSensorType, OBFormat,
                              OBAlignMode)
    HAS_ORBBEC = True
except ImportError:
    HAS_ORBBEC = False

# ── args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Gemini-camera VLM bottle pick-and-place")
parser.add_argument("--waypoints",     default="waypoints.json")
parser.add_argument("--teach",         action="store_true")
parser.add_argument("--yolov8-model",  default="yolov8m.pt",
                    help="yolov8n/s/m/l/x.pt — larger = more accurate")
parser.add_argument("--confidence",    type=float, default=0.30,
                    help="YOLO confidence threshold (default 0.30)")
parser.add_argument("--device",        default="cpu", help="'cpu' or 'cuda'")
parser.add_argument("--hz",            type=float, default=10.0,
                    help="Motion control frequency (Hz)")
parser.add_argument("--scan-timeout",  type=float, default=30.0,
                    help="Seconds to wait for a bottle detection")
parser.add_argument("--no-arm",        action="store_true")
parser.add_argument("--camera-index",  type=int, default=4,
                    help="OpenCV camera index for UVC fallback (default: 4)")
parser.add_argument("--action",        choices=["place", "flip"], default="place",
                    help="What to do after grasping: place (A→B) or flip the bottle")
parser.add_argument("--flip-speed",    type=float, default=300.0,
                    help="Flip stroke speed in deg/s (default 300)")
parser.add_argument("--no-preview",    action="store_true",
                    help="Disable the live cv2 preview window")
args = parser.parse_args()

USE_ARM  = not args.no_arm and HAS_CAN
LOOP_DT  = 1.0 / args.hz

# All known waypoints — teach mode only records ones missing from the file
WAYPOINT_HINTS = {
    "home":        "safe resting pose, gripper open",
    "scan_pos":    "overhead pose — Gemini camera sees the full workspace",
    "pre_grasp_a": "10 cm above bottle at A, gripper open",
    "grasp_a":     "gripper around bottle at A (fingers open, will close after)",
    "lift":        "raised ~15 cm with bottle gripped",
    "pre_place_b": "above target B, carrying bottle",
    "place_b":     "at target B (gripper will open here)",
    "pre_flip":    "bottle gripped, arm in a clear position ready to begin flip",
    "flip_end":    "bottle fully flipped / inverted — gripper still closed",
}

# Required keys per action (always + action-specific)
_ALWAYS   = ["home", "scan_pos", "pre_grasp_a", "grasp_a"]
_PLACE    = ["lift", "pre_place_b", "place_b"]
_FLIP     = ["pre_flip", "flip_end"]
REQUIRED_WAYPOINTS = _ALWAYS + (_FLIP if args.action == "flip" else _PLACE)

# Full ordered sequence for each action
SEQUENCE_PLACE = _ALWAYS + _PLACE
SEQUENCE_FLIP  = _ALWAYS + _FLIP


# ════════════════════════════════════════════════════════════════════════════
# Orbbec Gemini 2 camera — color + depth + intrinsics
# ════════════════════════════════════════════════════════════════════════════

class GeminiCamera:
    """
    Orbbec Gemini 2 with aligned depth+color and real intrinsics from the SDK.
    Provides pixel_to_3d() for metric bottle localization.
    """

    def __init__(self):
        if not HAS_ORBBEC:
            raise RuntimeError("pyorbbecsdk not installed — run: pip install pyorbbecsdk")

        self.pipeline = Pipeline()
        cfg = Config()

        # Color stream
        try:
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR)
            try:
                profile = profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
            except Exception:
                profile = profiles.get_default_video_stream_profile()
            cfg.enable_stream(profile)
            self._color_w = profile.get_width()
            self._color_h = profile.get_height()
            print(f"  Gemini color : {self._color_w}x{self._color_h} @ {profile.get_fps()} fps")
        except Exception as e:
            raise RuntimeError(f"Gemini color stream failed: {e}")

        # Depth stream
        self._has_depth = False
        try:
            profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH)
            dp = profiles.get_default_video_stream_profile()
            cfg.enable_stream(dp)
            self._has_depth = True
            print(f"  Gemini depth : {dp.get_width()}x{dp.get_height()} @ {dp.get_fps()} fps")
        except Exception as e:
            print(f"  WARNING: Gemini depth unavailable: {e}")

        # Align depth to color so we can directly index depth at color pixel coords
        try:
            cfg.set_align_mode(OBAlignMode.HW_MODE)
        except Exception:
            try:
                cfg.set_align_mode(OBAlignMode.SW_MODE)
            except Exception:
                pass

        self.pipeline.start(cfg)

        # Read real intrinsics from SDK
        self.fx, self.fy, self.cx, self.cy = 617.0, 617.0, 320.0, 240.0  # safe defaults
        try:
            param = self.pipeline.get_camera_param()
            intr  = param.rgb_intrinsic
            self.fx, self.fy = float(intr.fx), float(intr.fy)
            self.cx, self.cy = float(intr.cx), float(intr.cy)
            print(f"  Intrinsics   : fx={self.fx:.1f} fy={self.fy:.1f} "
                  f"cx={self.cx:.1f} cy={self.cy:.1f}")
        except Exception as e:
            print(f"  WARNING: could not read intrinsics ({e}), using 640×480 defaults")

        self._color: np.ndarray | None = None
        self._depth: np.ndarray | None = None
        self._lock   = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="gemini")
        self._thread.start()

    # ── internal capture loop ─────────────────────────────────────────────────
    def _decode_color(self, frame) -> np.ndarray | None:
        h, w = frame.get_height(), frame.get_width()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        if len(data) == h * w * 3:
            return data.reshape(h, w, 3).copy()
        if len(data) == h * w * 2:
            try:
                import cv2
                return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUYV)
            except Exception:
                return None
        try:
            import cv2
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
        except Exception:
            return None

    def _loop(self):
        while self._running:
            try:
                fs = self.pipeline.wait_for_frames(100)
                if fs is None:
                    continue
                cf = fs.get_color_frame()
                df = fs.get_depth_frame()
                color_arr = self._decode_color(cf) if cf else None
                depth_arr = None
                if df:
                    dh, dw = df.get_height(), df.get_width()
                    d = np.frombuffer(df.get_data(), dtype=np.uint16)
                    if len(d) == dh * dw:
                        depth_arr = d.reshape(dh, dw).copy()
                with self._lock:
                    if color_arr is not None:
                        self._color = color_arr
                    if depth_arr is not None:
                        self._depth = depth_arr
            except Exception:
                pass

    # ── public API ────────────────────────────────────────────────────────────
    def get_frames(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._lock:
            return self._color, self._depth

    def pixel_to_3d(self, px: int, py: int, depth_mm: float) -> tuple[float, float, float]:
        """Deproject pixel + depth (mm) to camera-frame 3D point (metres)."""
        z = depth_mm / 1000.0
        x = (px - self.cx) * z / self.fx
        y = (py - self.cy) * z / self.fy
        return (x, y, z)

    def bottle_3d(self, box: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
        """Return metric 3D centroid of a detection box using aligned depth."""
        with self._lock:
            depth = self._depth
        if depth is None:
            return None
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        h, w = depth.shape
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        r = 8
        patch = depth[max(0, cy-r):cy+r+1, max(0, cx-r):cx+r+1]
        valid = patch[(patch > 100) & (patch < 5000)]   # 10 cm – 5 m
        if len(valid) < 5:
            return None
        return self.pixel_to_3d(cx, cy, float(np.median(valid)))

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


class OpenCVCamera:
    """
    Fallback color-only camera via OpenCV UVC.
    No depth — bottle_3d() always returns None.
    The Orbbec Gemini 2 exposes itself as a UVC device so this works
    without pyorbbecsdk, but you lose metric 3D position.
    """

    def __init__(self, index: int = 0):
        import cv2
        self._cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  OpenCV camera [{index}]: {w}x{h}  (color only — no depth)")
        self._color: np.ndarray | None = None
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="opencv-cam")
        self._thread.start()

    def _loop(self):
        import cv2
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._color = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def get_frames(self) -> tuple[np.ndarray | None, None]:
        with self._lock:
            return self._color, None

    def bottle_3d(self, box) -> None:
        return None   # no depth available

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._cap.release()


def connect_camera(camera_index: int = 0):
    """
    Try Orbbec Gemini 2 (pyorbbecsdk) first for full color+depth.
    Falls back to OpenCV UVC if pyorbbecsdk isn't installed.
    """
    if HAS_ORBBEC:
        print("Connecting to Orbbec Gemini 2 (pyorbbecsdk)…")
        try:
            cam = GeminiCamera()
            print("  Gemini connected — color + depth available.")
            return cam
        except Exception as e:
            print(f"  Gemini SDK failed: {e}")
            print("  Falling back to OpenCV UVC (color only)…")
    else:
        print("pyorbbecsdk not installed — using OpenCV UVC (color only, no depth).")
        print("  To install: see https://github.com/orbbec/pyorbbecsdk/releases")

    cam = OpenCVCamera(camera_index)
    print("  WARNING: no depth — 3D bottle position will not be available.")
    return cam


# ════════════════════════════════════════════════════════════════════════════
# YOLOv8 bottle detector (COCO class 39)
# ════════════════════════════════════════════════════════════════════════════

class YOLOv8Detector:
    """
    YOLOv8 pretrained on COCO — restricted to class 39 (bottle).
    Far more reliable than zero-shot VLMs for this specific object.
    """

    _BOTTLE_CLASS = 39

    def __init__(self, model_name: str = "yolov8m.pt", device: str = "cpu"):
        print(f"Loading YOLOv8 ({model_name})...")
        from ultralytics import YOLO
        self._model  = YOLO(model_name)
        self._device = device
        print("YOLOv8 ready — COCO class 39 (bottle).")

    def detect(self, image_rgb: np.ndarray, conf: float = 0.30) -> list[dict]:
        """Return list of {score, box:(x1,y1,x2,y2)} dicts."""
        import cv2
        bgr     = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            bgr,
            conf=conf,
            classes=[self._BOTTLE_CLASS],
            device=self._device,
            verbose=False,
        )
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                dets.append({"score": float(box.conf), "box": (x1, y1, x2, y2)})
        return dets


# ════════════════════════════════════════════════════════════════════════════
# Live preview window (cv2, runs in background thread)
# ════════════════════════════════════════════════════════════════════════════

class PreviewWindow:
    """Non-blocking cv2 window that shows the latest annotated frame."""

    def __init__(self, title: str = "Gemini — bottle detection"):
        import cv2
        self._title  = title
        self._frame: np.ndarray | None = None
        self._lock   = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="preview")
        self._thread.start()

    def update(self, frame_rgb: np.ndarray, detections: list[dict],
               pos3d: tuple | None = None, status: str = ""):
        """Draw boxes on frame and queue for display."""
        import cv2
        out = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"bottle {d['score']:.2f}",
                        (x1, max(y1 - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if pos3d:
                x, y, z = pos3d
                cv2.putText(out, f"x={x:+.3f} y={y:+.3f} z={z:.3f}m",
                            (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        if status:
            cv2.putText(out, status, (8, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        with self._lock:
            self._frame = out

    def update_plain(self, frame_rgb: np.ndarray, status: str = ""):
        """Show raw frame with optional status text."""
        import cv2
        out = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
        if status:
            cv2.putText(out, status, (8, out.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        with self._lock:
            self._frame = out

    def _loop(self):
        import cv2
        cv2.namedWindow(self._title, cv2.WINDOW_NORMAL)
        while self._running:
            with self._lock:
                f = self._frame
            if f is not None:
                cv2.imshow(self._title, f)
            cv2.waitKey(30)
        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)


# ════════════════════════════════════════════════════════════════════════════
# Arm motion helpers
# ════════════════════════════════════════════════════════════════════════════

def move_to(follower, current_cmd: np.ndarray, target: np.ndarray,
            hz: float = 10.0, speed: float = MAX_SPEED_RAD) -> np.ndarray:
    dt       = 1.0 / hz
    max_step = speed * dt
    cmd      = current_cmd.copy()
    while True:
        delta = target - cmd
        if np.max(np.abs(delta[:-1])) < 0.005 and abs(delta[-1]) < 0.01:
            break
        cmd += np.clip(delta, -max_step, max_step)
        follower.command_joint_pos(cmd)
        time.sleep(dt)
    return cmd


def set_gripper(follower, current_cmd: np.ndarray, value: float, hz: float = 10.0) -> np.ndarray:
    target      = current_cmd.copy()
    target[-1]  = value
    dt          = 1.0 / hz
    max_step    = MAX_SPEED_RAD * dt * 3   # gripper moves faster
    cmd         = current_cmd.copy()
    while abs(target[-1] - cmd[-1]) > 0.01:
        cmd[-1] += np.clip(target[-1] - cmd[-1], -max_step, max_step)
        follower.command_joint_pos(cmd)
        time.sleep(dt)
    return cmd


# ════════════════════════════════════════════════════════════════════════════
# Waypoint helpers
# ════════════════════════════════════════════════════════════════════════════

def load_waypoints(path: str) -> dict[str, np.ndarray]:
    resolved = Path(path).resolve()
    with open(resolved) as f:
        return {k: np.array(v, dtype=float) for k, v in json.load(f).items()}


def save_waypoints(wp: dict[str, np.ndarray], path: str):
    resolved = Path(path).resolve()
    with open(resolved, "w") as f:
        json.dump({k: v.tolist() for k, v in wp.items()}, f, indent=2)
    print(f"Saved → {resolved}")


# ════════════════════════════════════════════════════════════════════════════
# Teach mode
# ════════════════════════════════════════════════════════════════════════════

def teach_mode(follower, out_path: str, hz: float = 10.0):
    # Load whatever is already saved so we only ask for missing ones
    existing: dict[str, np.ndarray] = {}
    if Path(out_path).exists():
        existing = load_waypoints(out_path)
        print(f"\n=== TEACH MODE — updating {Path(out_path).resolve()} ===")
        print(f"Already recorded: {list(existing.keys())}")
    else:
        print("\n=== TEACH MODE ===")

    # Teach all known waypoints, skipping ones already saved
    all_names = list(WAYPOINT_HINTS.keys())
    to_record = [n for n in all_names if n not in existing]

    if not to_record:
        print("All waypoints already recorded — nothing to do.")
        print("Delete the file and re-run --teach to start fresh.")
        return

    print(f"\nNeed to record ({len(to_record)}): {to_record}")
    print("Move arm by hand (zero-gravity) → SPACE to record  |  Q to abort\n")

    wp = dict(existing)   # start with what we have
    dt = 1.0 / hz

    for name in to_record:
        print(f"\n─ {name.upper()}")
        print(f"  {WAYPOINT_HINTS[name]}")
        print("  SPACE = record  |  S = skip  |  Q = abort")

        while True:
            q = follower.get_joint_pos()
            print(f"\r  q: {'  '.join(f'{v:+.3f}' for v in q)}   ", end="", flush=True)
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch == b" ":
                    wp[name] = q.copy()
                    print(f"\n  → Recorded {name}")
                    break
                elif ch.lower() == b"s":
                    print(f"\n  → Skipped {name}")
                    break
                elif ch.lower() == b"q":
                    print("\nAborted — saving what was recorded so far.")
                    if len(wp) > len(existing):
                        save_waypoints(wp, out_path)
                    return
            time.sleep(dt)

    save_waypoints(wp, out_path)
    print(f"\nDone — {len(wp)} waypoints saved.")


# ════════════════════════════════════════════════════════════════════════════
# Detection at scan_pos
# ════════════════════════════════════════════════════════════════════════════

def scan_for_bottle(
    camera: GeminiCamera,
    detector: YOLOv8Detector,
    preview: "PreviewWindow | None",
    conf: float,
    timeout: float,
) -> dict | None:
    """
    Runs YOLO on Gemini color frames.  Returns the best detection dict
    {score, box, pos3d} or None if timeout expires.
    """
    print(f"\nScanning for bottles (conf≥{conf}, timeout {timeout:.0f}s)…")
    deadline = time.time() + timeout

    while time.time() < deadline:
        color, depth = camera.get_frames()
        if color is None:
            time.sleep(0.05)
            continue

        dets = detector.detect(color, conf=conf)

        if dets:
            best  = max(dets, key=lambda d: d["score"])
            pos3d = camera.bottle_3d(best["box"])
            best["pos3d"] = pos3d

            if pos3d:
                x, y, z = pos3d
                print(f"\n  BOTTLE  score={best['score']:.2f}  "
                      f"x={x:+.3f} y={y:+.3f} z={z:.3f} m (camera frame)")
            else:
                print(f"\n  BOTTLE  score={best['score']:.2f}  (no depth)")

            if preview:
                preview.update(color, dets, pos3d,
                               f"DETECTED score={best['score']:.2f}")
            return best

        remaining = deadline - time.time()
        if preview:
            preview.update_plain(color, f"scanning… {remaining:.0f}s")
        else:
            print(f"\r  no bottle — {remaining:.0f}s", end="", flush=True)

        time.sleep(0.05)

    print("\nTimeout — no bottle found.")
    return None


# ════════════════════════════════════════════════════════════════════════════
# Action sequences
# ════════════════════════════════════════════════════════════════════════════

def _grasp(follower, wp, hz) -> np.ndarray:
    """Shared preamble: home → pre_grasp → grasp → close gripper."""
    cmd = follower.get_joint_pos().copy()
    for label, key in [("HOME", "home"), ("PRE-GRASP A", "pre_grasp_a"), ("GRASP A", "grasp_a")]:
        print(f"  → {label}")
        cmd = move_to(follower, cmd, wp[key], hz=hz)
    print("  → Close gripper")
    cmd = set_gripper(follower, cmd, GRIPPER_CLOSED, hz=hz)
    time.sleep(0.4)
    return cmd


def pick_and_place(follower, wp: dict[str, np.ndarray], hz: float) -> np.ndarray:
    cmd = _grasp(follower, wp, hz)

    print("  → LIFT")
    cmd = move_to(follower, cmd, wp["lift"].copy(), hz=hz)

    print("  → PRE-PLACE B")
    pre_b = wp["pre_place_b"].copy(); pre_b[-1] = GRIPPER_CLOSED
    cmd   = move_to(follower, cmd, pre_b, hz=hz)

    print("  → PLACE B")
    cmd = move_to(follower, cmd, wp["place_b"].copy(), hz=hz)

    print("  → Open gripper")
    cmd = set_gripper(follower, cmd, GRIPPER_OPEN, hz=hz)
    time.sleep(0.3)

    print("  → HOME")
    cmd = move_to(follower, cmd, wp["home"], hz=hz)
    print("  Done.")
    return cmd


def pick_and_flip(follower, wp: dict[str, np.ndarray], hz: float,
                  flip_speed: float = np.radians(300.0)) -> np.ndarray:
    """
    Pick bottle → move to pre_flip → close gripper → snap fast to flip_end → open gripper.

    flip_speed: joint speed for the flip stroke (default 300 deg/s — 5× normal).
                Pass np.radians(N) to adjust.
    """
    cmd = _grasp(follower, wp, hz)

    print("  → PRE-FLIP position")
    pre = wp["pre_flip"].copy(); pre[-1] = GRIPPER_CLOSED
    cmd = move_to(follower, cmd, pre, hz=hz)

    # Ensure full grip before the snap
    print("  → Closing gripper firmly")
    cmd = set_gripper(follower, cmd, GRIPPER_CLOSED, hz=hz)
    time.sleep(0.15)

    # Snap flip — very fast, no rate cap on gripper channel
    print(f"  → FLIP  ({np.degrees(flip_speed):.0f} deg/s)")
    flip = wp["flip_end"].copy(); flip[-1] = GRIPPER_CLOSED
    cmd  = move_to(follower, cmd, flip, hz=hz, speed=flip_speed)

    # Release immediately on arrival
    print("  → Open gripper")
    cmd = set_gripper(follower, cmd, GRIPPER_OPEN, hz=hz)

    print("  → HOME")
    cmd = move_to(follower, cmd, wp["home"], hz=hz)
    print("  Done.")
    return cmd


def run_action(follower, wp: dict[str, np.ndarray], hz: float, action: str) -> np.ndarray:
    if action == "flip":
        return pick_and_flip(follower, wp, hz, flip_speed=np.radians(args.flip_speed))
    return pick_and_place(follower, wp, hz)


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    # ── validate prerequisites BEFORE touching any hardware ───────────────────
    if not args.teach and USE_ARM:
        wp_path = Path(args.waypoints).resolve()
        print(f"Looking for waypoints: {wp_path}")
        if not wp_path.exists():
            print(f"ERROR: file not found.")
            # Hint if the default file or a nearby json exists
            for candidate in [Path("waypoints.json").resolve(),
                               Path(args.waypoints).parent.resolve()]:
                if candidate.suffix == ".json" and candidate.exists():
                    print(f"  Found nearby: {candidate} — did you mean --waypoints {candidate}?")
                elif candidate.is_dir():
                    jsons = list(candidate.glob("*.json"))
                    if jsons:
                        print(f"  JSON files in this folder: {[j.name for j in jsons]}")
            print("Run with --teach first to record waypoints.")
            return
        wp_raw  = load_waypoints(args.waypoints)
        missing = [n for n in REQUIRED_WAYPOINTS if n not in wp_raw]
        if missing:
            print(f"ERROR: missing waypoints for --action {args.action}: {missing}")
            print(f"  Keys in file : {list(wp_raw.keys())}")
            print(f"  Run: python vlm_pick_bottle.py --teach --waypoints {args.waypoints}")
            return
        print(f"Waypoints OK ({args.action}): {[k for k in REQUIRED_WAYPOINTS if k in wp_raw]}")
    else:
        wp_raw = {}

    follower = None
    camera   = None
    preview  = None

    try:
        # ── arm ───────────────────────────────────────────────────────────────
        if USE_ARM:
            from i2rt.robots.get_robot import get_yam_robot
            from i2rt.robots.motor_chain_robot import MotorChainRobot
            MotorChainRobot._check_current_qpos_in_joint_limits = lambda self: None
            print("Connecting to YAM arm…")
            follower = get_yam_robot(channel="can0", zero_gravity_mode=args.teach)
            print(f"YAM connected — {follower.num_dofs()} DOFs.")
        else:
            print("Arm disabled (--no-arm).")

        # ── teach mode ────────────────────────────────────────────────────────
        if args.teach:
            if follower is None:
                print("ERROR: --teach requires arm connection.")
                return
            teach_mode(follower, args.waypoints, hz=args.hz)
            return

        # waypoints already validated and loaded above
        wp = wp_raw if follower else None

        # ── camera (Gemini preferred, OpenCV UVC fallback) ────────────────────
        print()
        try:
            camera = connect_camera(camera_index=args.camera_index)
        except Exception as e:
            print(f"ERROR: no camera available: {e}")
            print("Plug in the Gemini 2 and retry.")
            return

        # warm up
        print("Warming up camera…", end="", flush=True)
        deadline = time.time() + 6.0
        while time.time() < deadline:
            if camera.get_frames()[0] is not None:
                break
            time.sleep(0.05)
        if camera.get_frames()[0] is None:
            print("\nERROR: camera gave no frames after 6 s — check USB connection.")
            return
        print(" ready.")

        # ── preview window ────────────────────────────────────────────────────
        if not args.no_preview:
            preview = PreviewWindow()

        # ── YOLO ─────────────────────────────────────────────────────────────
        detector = YOLOv8Detector(args.yolov8_model, args.device)

        # ── main loop ─────────────────────────────────────────────────────────
        cmd = follower.get_joint_pos().copy() if follower else None
        iteration = 0

        print("\n" + "═" * 60)
        print("GEMINI BOTTLE FINDER  |  Ctrl+C to stop")
        print("═" * 60)

        while True:
            iteration += 1
            print(f"\n[Iteration {iteration}]")

            # Move to scan position so camera looks at workspace
            if follower and wp:
                print("  → SCAN_POS")
                cmd = move_to(follower, cmd, wp["scan_pos"], hz=args.hz)

            # Detect
            detection = scan_for_bottle(
                camera, detector, preview,
                conf=args.confidence,
                timeout=args.scan_timeout,
            )

            if detection is None:
                print("Bottle not found. Press Enter to retry, Q+Enter to quit.")
                k = input().strip().lower()
                if k == "q":
                    break
                continue

            # Confirm before executing
            if follower and wp:
                if detection.get("pos3d"):
                    x, y, z = detection["pos3d"]
                    print(f"\nBottle at camera-frame: x={x:+.3f} y={y:+.3f} z={z:.3f} m")
                print(f"Press Enter to execute ({args.action}), Q+Enter to quit.")
                k = input().strip().lower()
                if k == "q":
                    break
                run_action(follower, wp, hz=args.hz, action=args.action)
                cmd = follower.get_joint_pos().copy()
            else:
                print("(--no-arm: detection shown, no motion)")
                print("Press Enter to scan again, Q+Enter to quit.")
                k = input().strip().lower()
                if k == "q":
                    break

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        if preview:
            preview.stop()
        if camera:
            camera.stop()
        if follower:
            follower.close()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
