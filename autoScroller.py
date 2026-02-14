import cv2
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

# Initialize client
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="Uoo3lejuM39FjoU4Lqiq"
)

# Configure video source (webcam)
source = WebcamSource(resolution=(1280, 720))

# Configure streaming options
config = StreamConfig(
    stream_output=["output_image"],  # Get video back with annotations
    data_output=["count_objects","predictions"],      # Get prediction data via datachannel
    requested_plan="webrtc-gpu-medium",  # Options: webrtc-gpu-small, webrtc-gpu-medium, webrtc-gpu-large
    requested_region="us",               # Options: us, eu, ap
    processing_timeout=600,              # 10 minutes
)

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize-15",
    workspace="emekas-workspace",
    image_input="image",
    config=config
)

# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

# Handle prediction data via datachannel
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")

# Run the session (blocks until closed)
session.run()


frame_h = None

@session.on_frame
def show_frame(frame, metadata):
    global frame_h
    frame_h = frame.shape[0]      # rows == pixel height
    cv2.imshow("Workflow Output", frame)
    ...

@session.on_data()
def on_data(data, metadata: VideoMetadata):
    # extract the y value from your payload (center y, ymin, etc.)
    y = ...  # adapt to your payload format

    if y is None: return

    # if y > 1 it's pixels, otherwise already normalized
    if y > 1 and frame_h:
        normalized = y / frame_h
    else:
        normalized = float(y)

    pct = int(round(max(0.0, min(1.0, normalized)) * 100))
    print("y(px)=", y, "normalized=", normalized, "pct=", pct)
    
## find y value range
## normalize from 0-1
## if y value <0.2, scroll down
# have y value correspond with brainrot percentage in index.html