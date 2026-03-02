import cv2
import json
import os
import time
import tempfile
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

LABEL_MIN_FRAMES = 5
LABEL_IMAGE_MAP = {
    "druski": "images/druski.png",
    "flight": "images/flight.png",
    "jimmy_butler": "images/jimmy_butler.png",
    "kai_cenat": "images/kai cenat.png",
    "speed_face": "images/Speed_face.png",
}

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

source = WebcamSource(resolution=(1280, 720))

config = StreamConfig(
    stream_output=["output_image"],
    data_output=["count_objects","predictions"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us"
)

session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize-23",
    workspace="emekas-workspace",
    image_input="image",
    config=config
)

@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    def _normalize_predictions(payload):
        if payload is None:
            return None

        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return None

        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict) and "predictions" in item:
                    payload = item
                    break

        if isinstance(payload, dict):
            predictions = payload.get("predictions")
        else:
            predictions = payload

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

    def _track_label(label_value):
        if not label_value:
            on_data._label_current = None
            on_data._label_count = 0
            on_data._label_stable = None
            return None, 0
        current = getattr(on_data, "_label_current", None)
        count = getattr(on_data, "_label_count", 0)
        if label_value == current:
            count += 1
        else:
            current = label_value
            count = 1
        stable = getattr(on_data, "_label_stable", None)
        if count >= LABEL_MIN_FRAMES:
            stable = label_value
        on_data._label_current = current
        on_data._label_count = count
        on_data._label_stable = stable
        return stable, count

    def _load_existing_output(path):
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    predictions = _normalize_predictions(data)
    label_value = _select_top_label(predictions)
    stable_label, label_frames = _track_label(label_value)
    label_image = LABEL_IMAGE_MAP.get(stable_label) if stable_label else None

    output_path = os.path.join(os.path.dirname(__file__), "meme.json")
    output = {
        "label": stable_label,
        "label_image": label_image,
        "label_frames": label_frames,
        "timestamp": int(time.time()),
    }
    
    temp_fd, temp_path = tempfile.mkstemp(dir=os.path.dirname(__file__), suffix='.json')
    try:
        with os.fdopen(temp_fd, 'w', encoding='utf-8') as handle:
            json.dump(output, handle)
        os.replace(temp_path, output_path)
    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass

session.run()
