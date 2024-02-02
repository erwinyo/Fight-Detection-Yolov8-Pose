import os
import threading

import cv2
import imutils
import torch
from flask import Flask, request, redirect
from flask import Response
import dotenv

import fight_module


# Load .env file
dotenv.load_dotenv()

YOLO_MODEL = os.getenv("YOLO_MODEL")
FIGHT_MODEL = os.getenv("FIGHT_MODEL")

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)


# Define a route for testing purposes
@app.route("/")
def index():
    return "This is a testing page. It tells you this is working :)"


# Route to check NVIDIA GPU information
@app.route("/nvidia")
def nvidia():
    gpu_info = ""

    if torch.cuda.is_available():
        gpu_info += "CUDA is available. Showing GPU information:\n"
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            gpu_info += f"> GPU {i} - Brand: {gpu.name}\n"

    return f"\n{gpu_info}\n"


# Route for raw video stream
@app.route("/raw")
def raw():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# Route to start processing video stream
@app.route("/start")
def start():
    video_input = request.args.get("video_input")

    # Start a thread that will perform motion detection
    t = threading.Thread(target=detect, args=(video_input,))
    t.daemon = True
    t.start()

    return redirect("/raw")


def detect(video_input):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock

    FIGHT_ON = False
    FIGHT_ON_TIMEOUT = 5  # second

    fdet = fight_module.FightDetector(FIGHT_MODEL)
    yolo = fight_module.YoloPoseEstimation(YOLO_MODEL)
    for result in yolo.estimate(video_input):

        # Get original image (without annotation, clean image) from YOLOv8
        orig_frame = result.orig_img
        # Get the result image from YOLOv8
        result_frame = result.plot()
        frame_height = result_frame.shape[0]
        frame_width = result_frame.shape[1]
        if result_frame.shape[0] > 720:
            result_frame = imutils.resize(result_frame, width=1280)

        try:
            boxes = result.boxes.xyxy.tolist()
            xyn = result.keypoints.xyn.tolist()
            confs = result.keypoints.conf
            ids = result.boxes.id

            confs = [] if confs is None else confs.tolist()
            ids = [] if ids is None else [str(int(ID)) for ID in ids]

            # Get interaction box
            interaction_boxes = fight_module.get_interaction_box(boxes)

            # Process only what is inside the interaction box
            for inter_box in interaction_boxes:
                # Green box
                cv2.rectangle(result_frame, (int(inter_box[0]), int(inter_box[1])),
                              (int(inter_box[2]), int(inter_box[3])), (0, 255, 0), 2)

                # Prediction starts here - per person - all person on the frame - including outside the interaction box
                both_fighting = []
                for conf, xyn, box, identity in zip(confs, xyn, boxes, ids):
                    # Check if the person is within the interaction box - filter only persons inside the interaction box
                    center_person_x, center_person_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
                    if inter_box[0] <= center_person_x <= inter_box[2] and inter_box[1] <= center_person_y <= \
                            inter_box[3]:
                        # Fight Detection
                        is_person_fighting = fdet.detect(conf, xyn)
                        both_fighting.append(is_person_fighting)

                else:
                    # Check if both fighting
                    if all(both_fighting):
                        FIGHT_ON = True

        except TypeError as te:
            pass
        except IndexError as ie:
            pass

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = result_frame.copy()

        # RING THE ALARM
        if FIGHT_ON:
            cv2.putText(result_frame, "FIGHTING", (int(inter_box[2]), int(inter_box[3])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)
            FIGHT_ON_TIMEOUT -= 0.1

        if FIGHT_ON_TIMEOUT <= 0:
            FIGHT_ON = False
            FIGHT_ON_TIMEOUT = 5


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    while True:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            continue

        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue

        # yield the output frame in the byte format
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'


# check to see if this is the main thread of execution
if __name__ == '__main__':
    """
        host : 0.0.0.0 
        - this is a must, cannot be changed to 127.0.0.1 
        - or it will cannot be accessed after been forwarded by docker to host IP

        port : 80 (up to you)
    """
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True, use_reloader=False)
