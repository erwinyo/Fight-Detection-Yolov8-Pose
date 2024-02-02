import base64
import threading

import cv2
import imutils
import torch
from flask import Flask, request, redirect
from flask import Response

import fight_module

YOLO_MODEL = "model/yolo/yolov8n-pose.pt"
FIGHT_MODEL = "model/fight/fight-model.pth"

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)


@app.route("/")
def index():
    return "This is testing page. It tell you this is working :)"


@app.route("/nvidia")
def nvidia():
    gpu_info = ""

    if torch.cuda.is_available():
        gpu_info += "CUDA is available. Showing GPU information:\n"
        for i in range(torch.cuda.device_count()):
            gpu = torch.cuda.get_device_properties(i)
            gpu_info += f"> GPU {i} - Brand: {gpu.name}\n"

    return f"\n{gpu_info}\n"


@app.route("/raw")
def raw():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


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

    FPS = 20
    FIGHT_ON = False
    FIGHT_ON_TIMEOUT = 20  # second

    fdet = fight_module.FightDetector(FIGHT_MODEL, FPS)
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

            # Process only what inside the interaction box
            for inter_box in interaction_boxes:
                # Green box
                cv2.rectangle(result_frame, (int(inter_box[0]), int(inter_box[1])),
                              (int(inter_box[2]), int(inter_box[3])), (0, 255, 0), 2)

                # Prediction start here - per person - all person on the frame - including outside the interaction box
                both_fighting = []
                for conf, xyn, box, identity in zip(confs, xyn, boxes, ids):
                    # Check if the person is within the interaction box - filter only person inside interaction box
                    center_person_x, center_person_y = (box[2] + box[0]) / 2, (box[3] + box[1]) / 2
                    if inter_box[0] <= center_person_x <= inter_box[2] and inter_box[1] <= center_person_y <= \
                            inter_box[3]:
                        # Fight Detection
                        is_person_fighting = fdet.detect(conf, xyn)
                        both_fighting.append(is_person_fighting)

                    # If fight occur then send cropped face to VMS
                    # For Face Recognition Task
                    if FIGHT_ON:
                        # Left side
                        right_shoulder_x = xyn[6][0]
                        right_ear_x = xyn[4][0]

                        # Right side
                        left_shoulder_x = xyn[5][0]
                        left_ear_x = xyn[3][0]

                        # Take the average of left and right shoulder Y-value
                        left_shoulder_y = xyn[5][1]
                        right_shoulder_y = xyn[6][1]

                        # Take nose Y-value
                        nose_y = xyn[0][1]

                        if (right_shoulder_x != 0 and right_ear_x != 0) \
                                and (left_shoulder_x != 0 and left_ear_x != 0) \
                                and (left_shoulder_y != 0 and right_shoulder_y != 0) \
                                and (nose_y != 0):

                            # Decide which one is the most fartest - shoulder or ear
                            x1 = int(min(right_shoulder_x, right_ear_x) * frame_width)
                            x2 = int(max(left_shoulder_x, left_ear_x) * frame_width)

                            avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                            # Calculate the distance with previous average value and nose Y-value
                            distance_nose_shoulder = abs(avg_shoulder_y - nose_y)
                            # Setting up Y coordinate
                            y1 = int((avg_shoulder_y - (distance_nose_shoulder * 2)) * frame_height)
                            y2 = int(avg_shoulder_y * frame_height)

                            # Negative coordinates not allowed
                            if x1 >= 0 and x2 >= 0 and y1 >= 0 and y2 >= 0:
                                width_face = x2 - x1
                                height_face = y2 - y1
                                print("CLING")
                                # Less than 96 not allowed
                                if width_face >= 96 and height_face >= 96:
                                    cropped_face = orig_frame[y1:y2, x1:x2]
                                    b64 = cv2.imencode('.jpg', cropped_face)[1]
                                    b64 = base64.b64encode(b64)
                                    b64 = str(b64)
                                    b64 = b64[2:-1]

                else:
                    # Check if both fighting
                    if all(both_fighting) or FIGHT_ON:
                        cv2.putText(result_frame, "FIGHTING", (int(inter_box[2]), int(inter_box[3])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
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
            print("RINGGGGGG")
            FIGHT_ON_TIMEOUT -= 1 / FPS

        if FIGHT_ON_TIMEOUT <= 0:
            FIGHT_ON = False
            FIGHT_ON_TIMEOUT = 20


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
    app.run(host="0.0.0.0", port=80, debug=True, threaded=True, use_reloader=False)
