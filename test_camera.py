"""
Quick camera test for Orbbec Gemini 2.
Tries pyorbbecsdk first, falls back to OpenCV UVC.
Shows feed in rerun + an OpenCV window.

Usage:
  python test_camera.py
  python test_camera.py --index 1   # try a different OpenCV camera index
"""

import argparse
import time
import numpy as np
import cv2
import rerun as rr

parser = argparse.ArgumentParser()
parser.add_argument("--index", type=int, default=None,
                    help="Force a specific OpenCV camera index (skips auto-detect)")
parser.add_argument("--list", action="store_true",
                    help="List all OpenCV camera indices that open and exit")
parser.add_argument("--no-rerun", action="store_true")
args = parser.parse_args()

# ── list mode ────────────────────────────────────────────────────────────────
if args.list:
    print("Scanning OpenCV camera indices 0-9...")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            cap.release()
            if ok:
                print(f"  [{i}] OK  shape={frame.shape}")
            else:
                print(f"  [{i}] opens but no frame")
        else:
            print(f"  [{i}] not found")
    raise SystemExit(0)

# ── rerun ─────────────────────────────────────────────────────────────────────
if not args.no_rerun:
    rr.init("camera_test", spawn=True)
    print("Rerun viewer launched.")

# ── try pyorbbecsdk ───────────────────────────────────────────────────────────
pipeline = None
try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
    print("pyorbbecsdk found — connecting to Orbbec Gemini 2...")
    pipeline = Pipeline()
    cfg = Config()

    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR)
    try:
        cp = color_profiles.get_video_stream_profile(640, 480, OBFormat.RGB, 30)
    except Exception:
        cp = color_profiles.get_default_video_stream_profile()
    cfg.enable_stream(cp)

    try:
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH)
        dp = depth_profiles.get_default_video_stream_profile()
        cfg.enable_stream(dp)
        has_depth = True
    except Exception:
        has_depth = False

    pipeline.start(cfg)
    print(f"Orbbec pipeline started  (color {cp.get_width()}x{cp.get_height()} "
          f"@ {cp.get_fps()} fps, depth={'yes' if has_depth else 'no'})")

    print("Press Q in the OpenCV window or Ctrl+C to quit.\n")
    while True:
        frameset = pipeline.wait_for_frames(200)
        if frameset is None:
            continue

        color_frame = frameset.get_color_frame()
        depth_frame = frameset.get_depth_frame() if has_depth else None

        color = None
        if color_frame:
            h, w = color_frame.get_height(), color_frame.get_width()
            data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            if len(data) == h * w * 3:
                color = data.reshape(h, w, 3)          # RGB
            else:
                color = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if color is not None:
                    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        depth = None
        if depth_frame:
            h, w = depth_frame.get_height(), depth_frame.get_width()
            data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            if len(data) == h * w:
                depth = data.reshape(h, w)

        if color is not None:
            cv2.imshow("Orbbec color (Q to quit)", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
            if not args.no_rerun:
                rr.set_time("wall_clock", timestamp=time.time())
                rr.log("camera/color", rr.Image(color))
        if depth is not None:
            depth_vis = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow("Orbbec depth", depth_vis)
            if not args.no_rerun:
                rr.log("camera/depth", rr.DepthImage(depth, meter=1000.0))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pipeline.stop()
    cv2.destroyAllWindows()
    raise SystemExit(0)

except ImportError:
    print("pyorbbecsdk not installed — falling back to OpenCV UVC.\n"
          "  To install: pip install pyorbbecsdk\n")
except Exception as e:
    if pipeline:
        try:
            pipeline.stop()
        except Exception:
            pass
    print(f"pyorbbecsdk error: {e}\nFalling back to OpenCV UVC.\n")

# ── OpenCV UVC fallback ───────────────────────────────────────────────────────
def try_index(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap

if args.index is not None:
    indices = [args.index]
else:
    print("Auto-detecting camera index (0-9)...")
    indices = list(range(10))

cap = None
for idx in indices:
    cap = try_index(idx)
    if cap:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Opened camera index {idx}  ({w}x{h})")
        break
    print(f"  index {idx}: not available")

if cap is None:
    print("\nNo camera found via OpenCV either.")
    print("Make sure the Orbbec Gemini 2 is connected and drivers are installed.")
    print("Try running with --list to enumerate all camera indices.")
    raise SystemExit(1)

print("Press Q in the OpenCV window or Ctrl+C to quit.\n")
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame — camera may have disconnected.")
            break
        cv2.imshow("Camera (Q to quit)", frame)
        if not args.no_rerun:
            rr.set_time("wall_clock", timestamp=time.time())
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rr.log("camera/color", rr.Image(rgb))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
