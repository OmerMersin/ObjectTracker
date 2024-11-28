import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

import threading
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class VideoStream(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(VideoStream, self).__init__(**properties)
        self.number_frames = 0
        self.fps = 15  # Adjust frame rate as needed
        self.duration = 1 / self.fps * Gst.SECOND  # Duration of a frame in nanoseconds
        self.launch_string = (
            'appsrc name=mysrc is-live=true block=true format=GST_FORMAT_TIME '
            'caps=video/x-raw,format=BGR,width=640,height=480,framerate={}/1 '
            '! videoconvert ! video/x-raw,format=I420 '
            '! x264enc speed-preset=ultrafast tune=zerolatency bitrate=1000 key-int-max=15 bframes=0 '
            '! rtph264pay name=pay0 pt=96'.format(self.fps)
        )

        # Initialize object detection variables
        self.init_detection()

        # Initialize the camera once
        self.cap = self.initialize_csi_camera()
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Failed to open CSI camera using libcamera.")

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

    def on_need_data(self, src, length):
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot read frame from camera")
            return

        # Process the frame (object detection and overlay)
        self.run_odt_and_draw_results(frame)

        # Create Gst.Buffer from frame
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = self.number_frames
        self.number_frames += 1

        retval = src.emit('push-buffer', buf)
        if retval != Gst.FlowReturn.OK:
            print('Failed to push buffer: {}'.format(retval))

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        appsrc = rtsp_media.get_element().get_child_by_name('mysrc')
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

    # Include your object detection methods here
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
        results = self.detect_objects(frame, threshold=threshold)

        height, width, _ = frame.shape

        for obj in results:
            ymin, xmin, ymax, xmax = obj['bounding_box']
            x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)

            class_id = obj['class_id']
            score = obj['score']
            label = f"{self.classes[class_id]}: {score * 100:.1f}%" if class_id < len(self.classes) else f"Unknown: {score * 100:.1f}%"

            color = [int(c) for c in self.COLORS[class_id % len(self.COLORS)]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def main():
    Gst.init(None)

    server = GstRtspServer.RTSPServer()
    server.set_service("8554")  # Default port

    factory = VideoStream()
    factory.set_shared(True)
    server.get_mount_points().add_factory("/mystream", factory)

    server.attach(None)

    print("RTSP streaming at rtsp://0.0.0.0:8554/mystream")

    loop = GLib.MainLoop()
    loop.run()

if __name__ == "__main__":
    main()
