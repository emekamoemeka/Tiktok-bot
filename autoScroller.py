import cv2
import asyncio
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata
import json
import os
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import pyautogui
except ImportError:
    pyautogui = None

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

SCROLL_TARGET = (-600, 600)
HEAD_MIN_Y = 80.0
HEAD_MAX_Y = 460.0
SCROLL_THRESHOLD = 0.8
SCROLL_COOLDOWN_SEC = 3.0
SCROLL_AMOUNT = -480
LABEL_MIN_FRAMES = 5
LABEL_IMAGE_MAP = {
    "druski": "images/druski.png",
    "flight": "images/flight.png",
    "jimmy_butler": "images/jimmy_butler.png",
    "kai_cenat": "images/kai cenat.png",
    "speed_face": "images/Speed_face.png",
}

STATE_HOST = "127.0.0.1"
STATE_PORT = 8765

_state_lock = threading.Lock()
_state_server = None
_state_server_thread = None
_state = {
    "value": 0,
    "percent": 0,
    "label": None,
    "label_image": None,
    "label_frames": 0,
    "timestamp": int(time.time()),
}


def _set_state(payload):
    with _state_lock:
        _state.update(payload)


def _get_state_snapshot():
    with _state_lock:
        return dict(_state)


class _StateHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/state"):
            self._send_json(_get_state_snapshot())
            return
        self._send_json({"error": "not_found"}, status=404)

    def log_message(self, format, *args):
        return


def _start_state_server():
    global _state_server, _state_server_thread
    if _state_server is not None:
        return

    _state_server = ThreadingHTTPServer((STATE_HOST, STATE_PORT), _StateHandler)
    _state_server_thread = threading.Thread(target=_state_server.serve_forever, daemon=True)
    _state_server_thread.start()
    print(f"State server running at http://{STATE_HOST}:{STATE_PORT}/state")


def _stop_state_server():
    global _state_server, _state_server_thread
    if _state_server is None:
        return
    try:
        _state_server.shutdown()
        _state_server.server_close()
    except Exception:
        pass
    if _state_server_thread is not None:
        _state_server_thread.join(timeout=1.0)
    _state_server = None
    _state_server_thread = None


def _cancel_pending_session_tasks(active_session):
    loop = getattr(active_session, "_loop", None)
    if loop is None or loop.is_closed() or not loop.is_running():
        return

    async def _cancel_all_tasks():
        current = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    try:
        asyncio.run_coroutine_threadsafe(_cancel_all_tasks(), loop).result(timeout=2.0)
    except Exception:
        pass

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

source = WebcamSource(resolution=(640, 480))

config = StreamConfig(
    stream_output=["output_image"],
    data_output=["count_objects","predictions"],
    requested_plan="webrtc-gpu-small",
    requested_region="us",
)

session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize-16",
    workspace="emekas-workspace",
    image_input="image",
    config=config
)

@session.on_frame
def show_frame(frame, metadata):
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Workflow Output", small_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

def _normalize_predictions(data):
    if data is None:
        return None

    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return None

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "predictions" in item:
                data = item
                break

    if isinstance(data, dict):
        predictions = data.get("predictions")
    else:
        predictions = data

    if isinstance(predictions, str):
        try:
            predictions = json.loads(predictions)
        except json.JSONDecodeError:
            return None

    if isinstance(predictions, dict):
        predictions = predictions.get("predictions") or predictions.get("detections")

    if not isinstance(predictions, list):
        return None

    return [p for p in predictions if isinstance(p, dict)]


def _normalize_label(label):
    if label is None:
        return None
    return str(label).strip().lower().replace(" ", "_")


def _select_top_label(items):
    if not items:
        return None
    best = max(items, key=lambda pred: pred.get("confidence", 0.0))
    return _normalize_label(best.get("class"))


def _track_label(on_data_fn, label_value):
    if not label_value:
        on_data_fn._label_current = None
        on_data_fn._label_count = 0
        on_data_fn._label_stable = None
        return None, 0
    current = getattr(on_data_fn, "_label_current", None)
    count = getattr(on_data_fn, "_label_count", 0)
    if label_value == current:
        count += 1
    else:
        current = label_value
        count = 1
    stable = getattr(on_data_fn, "_label_stable", None)
    if count >= LABEL_MIN_FRAMES:
        stable = label_value
    on_data_fn._label_current = current
    on_data_fn._label_count = count
    on_data_fn._label_stable = stable
    return stable, count


def _get_head_y_position(predictions):
    if not predictions:
        return None

    head_predictions = [
        p for p in predictions
        if "head" in str(p.get("class", "")).lower()
    ]
    if not head_predictions:
        return None

    best = max(head_predictions, key=lambda p: p.get("confidence", 0))
    return best.get("y")


def _normalize_head_y(head_y, min_y=HEAD_MIN_Y, max_y=HEAD_MAX_Y):
    if head_y is None:
        return None

    try:
        y = float(head_y)
    except (TypeError, ValueError):
        return None

    if y <= min_y:
        return 0.0
    if y >= max_y:
        return 1.0

    return (y - min_y) / (max_y - min_y)


def _safe_move_and_scroll():
    if not pyautogui:
        return False

    try:
        if os.name == "nt":
            import ctypes

            user32 = ctypes.windll.user32
            left = user32.GetSystemMetrics(76)
            top = user32.GetSystemMetrics(77)
            width = user32.GetSystemMetrics(78)
            height = user32.GetSystemMetrics(79)
            right = left + width - 1
            bottom = top + height - 1
        else:
            width, height = pyautogui.size()
            left, top = 0, 0
            right, bottom = width - 1, height - 1

        target_x = max(left + 1, min(right - 1, SCROLL_TARGET[0]))
        target_y = max(top + 1, min(bottom - 1, SCROLL_TARGET[1]))
        pyautogui.moveTo(target_x, target_y, duration=0.05)
        pyautogui.scroll(SCROLL_AMOUNT)
        return True
    except pyautogui.FailSafeException:
        return False


@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    now = time.monotonic()

    predictions = _normalize_predictions(data)
    head_y = _get_head_y_position(predictions)
    normalized = _normalize_head_y(head_y)
    
    if normalized is not None:
        on_data._last_normalized = normalized
    else:
        normalized = getattr(on_data, "_last_normalized", None)

    label_value = _select_top_label(predictions)
    stable_label, label_frames = _track_label(on_data, label_value)
    label_image = LABEL_IMAGE_MAP.get(stable_label) if stable_label else None

    percent = int(round(normalized * 100.0)) if normalized is not None else 0

    last_scroll = getattr(on_data, "_last_scroll", 0.0)
    was_above = getattr(on_data, "_was_above", False)
    is_above = (normalized >= SCROLL_THRESHOLD) if normalized is not None else False
    if pyautogui and is_above and not was_above:
        if now - last_scroll >= SCROLL_COOLDOWN_SEC:
            if _safe_move_and_scroll():
                on_data._last_scroll = now
                print(f"head_y_norm={normalized:.3f}")
    on_data._was_above = is_above

    output = {
        "value": normalized if normalized is not None else 0,
        "percent": percent,
        "label": stable_label,
        "label_image": label_image,
        "label_frames": label_frames,
        "timestamp": int(time.time()),
    }
    _set_state(output)

_start_state_server()
try:
    session.run()
except KeyboardInterrupt:
    print("Stopped by user.")
except RuntimeError as exc:
    message = str(exc)
    is_credit_error = ("HTTP 402" in message) or ("CreditsExceededError" in message)
    if is_credit_error:
        _set_state({
            "value": 0,
            "percent": 0,
            "label": None,
            "label_image": None,
            "label_frames": 0,
            "timestamp": int(time.time()),
            "error": "roboflow_credits_exhausted"
        })
        print("Roboflow credits are exhausted (HTTP 402).")
        print("Top up credits or switch to a local/free model endpoint, then run again.")
    else:
        raise
finally:
    _cancel_pending_session_tasks(session)
    try:
        session.close()
    except Exception:
        pass
    _stop_state_server()
    cv2.destroyAllWindows()
