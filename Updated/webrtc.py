import argparse
import asyncio
import cv2
import numpy as np
import json
from fractions import Fraction
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from tflite_runtime.interpreter import Interpreter
import time

class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that captures video from the Raspberry Pi CSI camera,
    runs object detection, and sends frames via WebRTC.
    """
    def __init__(self):
        super().__init__()
        # Use GStreamer pipeline to capture from the CSI camera
        self.cap = self.initialize_csi_camera()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError('Could not open CSI camera')

        # Set camera properties
        # Note: These settings might not have an effect when using GStreamer pipelines
        # You can adjust width, height, and framerate in the GStreamer pipeline directly

        self.init_detection()
        self.start_time = time.time()
        self.frame_count = 0

        # Detection interval to reduce CPU usage
        self.detection_interval = 5  # Run object detection every 5 frames

    def initialize_csi_camera(self):
        # GStreamer pipeline for libcamera on Raspberry Pi
        gst_pipeline = (
            "libcamerasrc ! "
            "video/x-raw,width=640,height=480,format=NV12,framerate=30/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        return cap

    def init_detection(self):
        # Load class labels (COCO labels)
        self.classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
                        "truck", "boat", "traffic light", "fire hydrant", "???(ID12)", "stop sign",
                        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                        "elephant", "bear", "zebra", "giraffe", "???(ID26)", "backpack", "umbrella",
                        "???(ID29)", "???(ID30)", "handbag", "tie", "suitcase", "frisbee", "skis",
                        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                        "skateboard", "surfboard", "tennis racket", "bottle", "???(ID45)", "wine glass",
                        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                        "chair", "couch", "potted plant", "bed", "???(ID66)", "dining table",
                        "???(ID68)", "???(ID69)", "toilet", "???(ID71)", "tv", "laptop", "mouse", "remote",
                        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                        "refrigerator", "???(ID83)", "book", "clock", "vase", "scissors",
                        "teddy bear", "hair drier", "toothbrush"]

        # Assign random colors for each class
        self.COLORS = np.random.randint(0, 255, size=(len(self.classes), 3), dtype=np.uint8)

        # Initialize TensorFlow Lite interpreter
        self.interpreter = Interpreter(model_path="ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Initialize tracking variables
        self.tracking = False
        self.tracker = None

    def set_input_tensor(self, image):
        tensor_index = self.input_details[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def get_output_tensor(self, index):
        output_details = self.output_details[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, image, threshold=0.5):
        input_shape = self.input_details[0]['shape'][1:3]
        resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
        input_data = np.expand_dims(resized_image, axis=0)

        if self.input_details[0]['dtype'] == np.uint8:
            input_data = input_data.astype(np.uint8)
        else:
            input_data = input_data.astype(np.float32) / 255.0

        self.set_input_tensor(input_data)
        self.interpreter.invoke()

        boxes = self.get_output_tensor(0)
        class_ids = self.get_output_tensor(1).astype(int)
        scores = self.get_output_tensor(2)
        count = int(self.get_output_tensor(3))

        results = []
        for i in range(count):
            if scores[i] >= threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': class_ids[i],
                    'score': scores[i]
                }
                results.append(result)
        return results

    def run_odt_and_draw_results(self, frame, threshold=0.5):
        detections = self.detect_objects(frame, threshold=threshold)

        height, width, _ = frame.shape

        for obj in detections:
            ymin, xmin, ymax, xmax = obj['bounding_box']
            x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

            class_id = obj['class_id']
            score = obj['score']
            label = f"{self.classes[class_id]}: {score * 100:.1f}%" if class_id < len(self.classes) else f"Unknown: {score * 100:.1f}%"

            color = [int(c) for c in self.COLORS[class_id % len(self.COLORS)]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run_tracking(self, frame):
        # Implement tracking logic if needed
        pass

    async def recv(self):
        # Get the next frame
        frame = await self.next_frame()
        print(f"Sending frame: {frame}")
        return frame

    async def next_frame(self):
        loop = asyncio.get_event_loop()
        ret, frame = await loop.run_in_executor(None, self.cap.read)
        if not ret:
            raise Exception("Failed to read frame from camera")    

        # Process frame
        self.process_frame(frame)

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create a VideoFrame
        video_frame = VideoFrame.from_ndarray(frame_rgb, format="rgb24")

        # Set timestamps
        now = time.time()
        video_frame.pts = int((now - self.start_time) * 90000)
        video_frame.time_base = Fraction(1, 90000)

        print(f"Frame ready: pts={video_frame.pts}")
        return video_frame


    def process_frame(self, frame):
        try:
            print("Processing frame...")
            self.frame_count += 1
            if self.frame_count % self.detection_interval == 0:
                self.run_odt_and_draw_results(frame)
            else:
                print("Skipping detection for this frame")
        except Exception as e:
            print(f"Error in process_frame: {e}")



async def index(request):
    content = open('index.html', 'r').read()
    return web.Response(content_type='text/html', text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Add local media
    video_track = VideoTransformTrack()
    pc.addTrack(video_track)

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        print('ICE connection state is %s' % pc.iceConnectionState)
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    # Handle offer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type='application/json',
        text=json.dumps({'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type})
    )

pcs = set()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WebRTC video streaming with object detection')
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen on')
    parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    args = parser.parse_args()

    # Create web app
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_post('/offer', offer)

    # Run web app
    web.run_app(app, host=args.host, port=args.port)
