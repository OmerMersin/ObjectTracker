from flask import Flask, Response, request, jsonify, render_template
import cv2
import threading
import numpy as np

app = Flask(__name__)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Shared variables for tracking
tracking_coords = None
lock = threading.Lock()
tracker = None
frame = None

def generate_frames():
    global frame, tracker, tracking_coords

    while True:
        success, frame = camera.read()  # Read the camera frame
        if not success:
            continue  # Skip if frame is not captured

        with lock:
            # Update the tracker if initialized
            if tracker is not None:
                try:
                    success, tracking_coords = tracker.update(frame)
                    if success:
                        x, y, w, h = map(int, tracking_coords)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                except cv2.error as e:
                    print(f"Tracker update error: {e}")
                    tracker = None  # Reset the tracker if it fails

        # Create a copy of the frame for streaming
        frame_for_streaming = frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame_for_streaming)
        if not ret:
            continue  # Skip if encoding fails

        # Yield the encoded frame as part of the HTTP response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')




@app.route('/video_feed')
def video_feed():
    """
    Route to stream video frames.
    """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_tracking', methods=['POST'])
def start_tracking():
    global tracker, tracking_coords, frame
    data = request.json
    if not data or 'x' not in data or 'y' not in data or 'w' not in data or 'h' not in data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        x = int(data['x'])
        y = int(data['y'])
        w = int(data['w'])
        h = int(data['h'])
    except ValueError:
        return jsonify({"error": "Coordinates must be integers"}), 400

    with lock:
        # Ensure the frame is not empty before initializing the tracker
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return jsonify({"error": "Frame is not available or invalid"}), 500

        # Initialize the tracker
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x, y, w, h))
        tracking_coords = (x, y, w, h)

    return jsonify({"status": "Tracking started", "coords": tracking_coords})


@app.route('/smart_brush', methods=['POST'])
def smart_brush():
    global tracker, tracking_coords, frame
    data = request.json
    if not data or 'x' not in data or 'y' not in data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        x = int(data['x'])
        y = int(data['y'])
    except ValueError:
        return jsonify({"error": "Coordinates must be integers"}), 400

    with lock:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return jsonify({"error": "Frame is not available or invalid"}), 500

        # Downsample the frame for faster processing
        scale = 0.5
        resized_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Adjust clicked coordinates for the scaled frame
        scaled_x = int(x * scale)
        scaled_y = int(y * scale)

        # Create a small region of interest (ROI) around the clicked point
        roi_size = 50
        x1 = max(0, scaled_x - roi_size)
        y1 = max(0, scaled_y - roi_size)
        x2 = min(resized_frame.shape[1], scaled_x + roi_size)
        y2 = min(resized_frame.shape[0], scaled_y + roi_size)
        roi = resized_frame[y1:y2, x1:x2]

        # Convert ROI to grayscale for edge and corner detection
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray_roi, 50, 150)

        # Find contours in the edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return jsonify({"error": "No contours found"}), 404

        # Find the largest contour (assuming it's the object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Scale back the coordinates to the original frame
        x = int((x1 + x) / scale)
        y = int((y1 + y) / scale)
        w = int(w / scale)
        h = int(h / scale)

        # Initialize the tracker with the bounding box
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x, y, w, h))
        tracking_coords = (x, y, w, h)

    return jsonify({"status": "Tracking started", "coords": tracking_coords})




@app.route('/')
def index():
    """
    Render the main page for video streaming and object selection.
    """
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
