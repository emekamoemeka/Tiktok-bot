import cv2
import time
import asyncio
import pyautogui
import os
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

last_print_time = {}

consecutive_count = {}
last_detected_gesture = None

CLICK_COORDS = {
    "thumbs_up": (-153, 1063),
    "up": (-153, 1063),
    "thumbs_down": (-150, 1195),
    "down": (-150, 1195),
    "peace": (-139, 1313),
}

def _perform_click(gesture):
    """Perform a desktop click for the given gesture."""
    if gesture not in CLICK_COORDS:
        print(f"No click coordinates configured for gesture: {gesture}")
        return False

    x, y = CLICK_COORDS[gesture]
    try:
        pyautogui.click(x=x, y=y)
        print(f"Clicked ({x}, {y}) for {gesture} gesture")
        return True
    except Exception as error:
        print(f"Click failed at ({x}, {y}) for {gesture}: {error}")
        return False


def _async_exception_handler(loop, context):
    exc = context.get("exception")
    handle = context.get("handle")
    if isinstance(exc, asyncio.InvalidStateError) and handle is not None:
        if "Transaction.__retry" in repr(handle):
            return
    loop.default_exception_handler(context)


try:
    asyncio.get_event_loop().set_exception_handler(_async_exception_handler)
except RuntimeError:
    pass

client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY
)

source = WebcamSource(resolution=(1280, 720))

config = StreamConfig(
    stream_output=["output_image"],
    data_output=["count_objects","predictions"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
)

session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize-21",
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
    global last_detected_gesture
    current_time = time.time()
    
    if 'predictions' in data and 'predictions' in data['predictions']:
        predictions = data['predictions']['predictions']
        
        if predictions:
            detected_gesture = None
            for prediction in predictions:
                if 'class' in prediction:
                    detected_gesture = prediction['class']
                    break
            
            if detected_gesture:
                if detected_gesture == last_detected_gesture:
                    consecutive_count[detected_gesture] = consecutive_count.get(detected_gesture, 0) + 1
                else:
                    consecutive_count.clear()
                    consecutive_count[detected_gesture] = 1
                    last_detected_gesture = detected_gesture
                
                if consecutive_count[detected_gesture] >= 3:
                    if detected_gesture not in last_print_time or (current_time - last_print_time[detected_gesture]) >= 5.0:
                        print(f"Detected: {detected_gesture} (3 consecutive ticks)")
                        
                        if detected_gesture in CLICK_COORDS:
                            _perform_click(detected_gesture)
                        
                        last_print_time[detected_gesture] = current_time
                        consecutive_count[detected_gesture] = 0
        else:
            consecutive_count.clear()
            last_detected_gesture = None
    else:
        consecutive_count.clear()
        last_detected_gesture = None

session.run()
