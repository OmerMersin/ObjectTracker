import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib, GstRtsp

from flask import Flask, request, jsonify
import threading
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import time

app = Flask(__name__)

class VideoStream(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(VideoStream, self).__init__(**properties)
        self.number_frames = 0
        self.fps = 15  # Adjust frame rate as needed
        self.duration = int(1 / self.fps * Gst.SECOND)  # Duration of a frame in nanoseconds

        # Use hardware-accelerated encoder
        self.launch_string = (
            'appsrc name=mysrc is-live=true block=true format=GST_FORMAT_TIME '
            'caps=video/x-raw,format=BGR,width=640,height=480,framerate={fps}/1 '
            '! videoconvert ! video/x-raw,format=I420 '
            '! x264enc speed-preset=ultrafast tune=zerolatency bitrate=1000 key-int-max=15 bframes=0 '
            '! rtph264pay name=pay0 pt=96'.format(fps=self.fps)
        )

        # Initialize object detection variables
        self.init_detection()

        # Initialize the camera once
        self.cap = self.initialize_csi_camera()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open CSI camera using libcamera.")

        # Initialize tracking variables
        self.tracker = None
        self.tracking = False
        self.track_box = None
        self.track_class_id = None
        self.lock = threading.Lock()

        # Shared frame buffer
        self.latest_frame = None

        # Detection interval
        self.detection_interval = 5  # Run object detection every 5 frames


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

        # OpenCV Object Tracker
        self.OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            # Add other trackers if needed
        }

    def on_need_data(self, src, length):
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                print("Cannot read frame from camera")
                return

            start_time = time.time()

            # Process the frame (object detection and overlay)
            if self.number_frames % self.detection_interval == 0:
                self.run_odt_and_draw_results(frame)
            else:
                self.run_tracking(frame)

            # Store the latest frame for Flask endpoint
            self.latest_frame = frame.copy()

            # Create Gst.Buffer from frame
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)

            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = self.number_frames
            self.number_frames += 1

            end_time = time.time()
            processing_time = end_time - start_time
            # Uncomment the following line to see processing time per frame
            # print(f"Frame processing time: {processing_time:.3f} seconds")

        retval = src.emit('push-buffer', buf)
        if retval != Gst.FlowReturn.OK:
            print('Failed to push buffer: {}'.format(retval))

    def run_tracking(self, frame):
        # Handle tracking
        if self.tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Add label
                if self.track_class_id is not None and 0 <= self.track_class_id < len(self.classes):
                    label = self.classes[self.track_class_id]
                else:
                    label = "Object"  # Default label
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Tracking failed
                self.tracking = False
                self.tracker = None

    # [Rest of your existing methods: run_odt_and_draw_results, detect_objects, etc.]

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.pipeline = rtsp_media.get_element()
        appsrc = self.pipeline.get_child_by_name('mysrc')
        appsrc.set_property('do-timestamp', True)
        appsrc.set_property('is-live', True)
        appsrc.set_property('format', Gst.Format.TIME)
        appsrc.connect('need-data', self.on_need_data)

    def initialize_csi_camera(self):
        # GStreamer pipeline for libcamera on Raspberry Pi
        gst_str = (
            "libcamerasrc ! "
            "video/x-raw, width=640, height=480, format=NV12 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not cam.isOpened():
            print("Failed to open CSI camera using libcamera.")
            return None
        return cam

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

        # Always run detection to get the latest detections
        for obj in detections:
            ymin, xmin, ymax, xmax = obj['bounding_box']
            x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

            class_id = obj['class_id']
            score = obj['score']
            label = f"{self.classes[class_id]}: {score * 100:.1f}%" if class_id < len(self.classes) else f"Unknown: {score * 100:.1f}%"

            color = [int(c) for c in self.COLORS[class_id % len(self.COLORS)]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Handle tracking
        if self.tracking and self.tracker is not None:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Add label
                if self.track_class_id is not None and 0 <= self.track_class_id < len(self.classes):
                    label = self.classes[self.track_class_id]
                else:
                    label = "Object"  # Default label
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                # Tracking failed, attempt re-detection
                self.tracking = False
                self.tracker = None
                # Try to find the object again
                for det in detections:
                    if det['class_id'] == self.track_class_id:
                        self.tracker = self.OPENCV_OBJECT_TRACKERS['csrt']()
                        bbox = det['box']
                        self.tracker.init(frame, bbox)
                        self.tracking = True
                        break
        else:
            # Attempt to find the object in detections
            for det in detections:
                if det['class_id'] == self.track_class_id:
                    self.tracker = self.OPENCV_OBJECT_TRACKERS['csrt']()
                    bbox = det['box']
                    self.tracker.init(frame, bbox)
                    self.tracking = True
                    break

# Flask route to select object
@app.route('/select_object', methods=['POST'])
def select_object():
    data = request.get_json()
    x = data['x']
    y = data['y']
    print(f"Received coordinates: x={x}, y={y}")

    # Access the VideoStream instance
    factory = app.config['factory']

    with factory.lock:
        # Use the latest frame from the RTSP server
        if factory.latest_frame is None:
            return jsonify({'status': 'failed', 'reason': 'No frame available'}), 500

        current_frame = factory.latest_frame.copy()
        detections = factory.detect_objects(current_frame, threshold=0.5)

        height, width, _ = current_frame.shape

        # First, try to find the object in detections
        for det in detections:
            ymin, xmin, ymax, xmax = det['bounding_box']
            x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
            if x1 <= x <= x2 and y1 <= y <= y2:
                factory.tracker = factory.OPENCV_OBJECT_TRACKERS['csrt']()
                bbox = (x1, y1, x2 - x1, y2 - y1)
                factory.tracker.init(current_frame, bbox)
                factory.tracking = True
                factory.track_class_id = det['class_id']
                return jsonify({'status': 'tracking started with detection'})

        # If no detection, use OpenCV methods (e.g., contour detection)
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edged = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find contour that contains the click point
        for cnt in contours:
            if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                x1, y1, w, h = cv2.boundingRect(cnt)
                factory.tracker = factory.OPENCV_OBJECT_TRACKERS['csrt']()
                factory.tracker.init(current_frame, (x1, y1, w, h))
                factory.tracking = True
                factory.track_class_id = None  # Unknown class
                return jsonify({'status': 'tracking started with contour'})

        return jsonify({'status': 'no object found'})

def start_rtsp_server():
    Gst.init(None)

    server = GstRtspServer.RTSPServer()
    server.set_service("8554")  # Default port

    factory = VideoStream()
    factory.set_shared(True)
    factory.set_protocols(GstRtsp.RTSPLowerTrans.UDP)
    server.get_mount_points().add_factory("/mystream", factory)

    # Store the factory in the Flask app config for access in the endpoint
    app.config['factory'] = factory

    server.attach(None)

    print("RTSP streaming at rtsp://0.0.0.0:8554/mystream")

    loop = GLib.MainLoop()
    loop.run()

if __name__ == '__main__':
    # Start RTSP server in a separate thread
    rtsp_thread = threading.Thread(target=start_rtsp_server)
    rtsp_thread.daemon = True
    rtsp_thread.start()

    # Start Flask app
    app.run(host='0.0.0.0', port=5000, threaded=True)
