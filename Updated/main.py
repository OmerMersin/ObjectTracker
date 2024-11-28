from flask import Flask, Response, request, render_template, jsonify
import cv2
import threading
import numpy as np
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

def initialize_csi_camera():
    # GStreamer pipeline for libcamera on Raspberry Pi
    gst_str = (
    "libcamerasrc ! "
    "video/x-raw, width=640, height=480, format=NV12 ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
    )


    # Open the camera using GStreamer pipeline
    cam = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if not cam.isOpened():
        raise RuntimeError("Failed to open CSI camera using libcamera.")
    return cam

# Initialize the camera
try:
    camera = initialize_csi_camera()
    print("Using CSI camera.")
except RuntimeError as e:
    print(f"Error: {e}")
    camera = None


frame = None
lock = threading.Lock()

# Load class labels (COCO labels)
classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
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
COLORS = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

# Initialize TensorFlow Lite interpreter
interpreter = Interpreter(model_path="ssd_mobilenet_v3_small_coco_2020_01_14/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# OpenCV Object Tracker
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    #"mosse": cv2.TrackerMOSSE_create,
    # Add other trackers if needed
}

tracker = None
tracking = False
track_box = None
track_class_id = None  # To keep track of the object's class

# Set input tensor
def set_input_tensor(interpreter, image):
    tensor_index = input_details[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

# Get output tensor
def get_output_tensor(interpreter, index):
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

# Detect objects
def detect_objects(image, threshold=0.5):
    input_shape = input_details[0]['shape'][1:3]
    resized_image = cv2.resize(image, (input_shape[1], input_shape[0]))
    input_data = np.expand_dims(resized_image, axis=0)

    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = input_data.astype(np.float32) / 255.0

    set_input_tensor(interpreter, input_data)
    interpreter.invoke()

    boxes = get_output_tensor(interpreter, 0)
    class_ids = get_output_tensor(interpreter, 1).astype(int)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

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

# Run detection and overlay results on frame
def run_odt_and_draw_results(frame, threshold=0.5):
    results = detect_objects(frame, threshold=threshold)

    height, width, _ = frame.shape
    detections = []

    for obj in results:
        ymin, xmin, ymax, xmax = obj['bounding_box']
        x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

        class_id = obj['class_id']
        score = obj['score']
        label = f"{classes[class_id]}: {score * 100:.1f}%" if class_id < len(classes) else f"Unknown: {score * 100:.1f}%"

        color = [int(c) for c in COLORS[class_id % len(COLORS)]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        detections.append({
            'box': (x1, y1, x2 - x1, y2 - y1),
            'class_id': class_id,
            'score': score
        })

    return detections

# Generate video frames
def generate_frames():
    global frame, tracking, tracker, track_box, track_class_id

    while True:
        success, frame = camera.read()
        if not success:
            continue

        with lock:
            # Always run detection to get the latest detections
            detections = run_odt_and_draw_results(frame, threshold=0.5)

            if tracking and tracker is not None:
                # Update the tracker
                success, box = tracker.update(frame)
                if success:
                    x, y, w, h = map(int, box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # Add label
                    if track_class_id is not None and 0 <= track_class_id < len(classes):
                        label = classes[track_class_id]
                    else:
                        label = "Object"  # Default label
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    # Tracking failed, attempt re-detection
                    tracking = False
                    tracker = None
                    # Try to find the object again
                    for det in detections:
                        if det['class_id'] == track_class_id:
                            tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                            tracker.init(frame, det['box'])
                            tracking = True
                            break
            else:
                # Attempt to find the object in detections
                for det in detections:
                    if det['class_id'] == track_class_id:
                        tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                        tracker.init(frame, det['box'])
                        tracking = True
                        break

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_object', methods=['POST'])
def select_object():
    global tracking, tracker, track_box, track_class_id
    data = request.get_json()
    x = data['x']
    y = data['y']
    print(x, y)

    with lock:
        current_frame = frame.copy()
        detections = run_odt_and_draw_results(current_frame, threshold=0.5)

        # First, try to find the object in detections
        for det in detections:
            x1, y1, w, h = det['box']
            x2, y2 = x1 + w, y1 + h
            if x1 <= x <= x2 and y1 <= y <= y2:
                tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                tracker.init(current_frame, det['box'])
                tracking = True
                track_class_id = det['class_id']
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
                tracker = OPENCV_OBJECT_TRACKERS['csrt']()
                tracker.init(current_frame, (x1, y1, w, h))
                tracking = True
                track_class_id = None  # Unknown class
                return jsonify({'status': 'tracking started with contour'})

        return jsonify({'status': 'no object found'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Run in threaded mode to handle multiple requests
    app.run(host='0.0.0.0', port=5000, threaded=True)
