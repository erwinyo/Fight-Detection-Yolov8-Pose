
# Fight Detection YOLOv8-Pose

ML solution to detect fight that occur using camera streaming.

## Acknowledgements

 - [IMSOO Git Repo](https://github.com/imsoo/fight_detection)
 - [YOLOv8 Pose Estimation](https://docs.ultralytics.com/tasks/pose/)



## Installation

#### REQUIREMENT

Install ananconda environment

Create new environment with python 3.10.12

```bash
conda create -n fdet python=3.10.12 pip

```
Enter the environment

```bash
conda activate fdet
```

#### RUNNING SERVER

Clone the repo, enter the directory

```bash
git clone https://github.com/erwinyo/Fight-Detection-Yolov8-Pose

cd Fight-Detection-Yolov8-Pose

```

Install the requirements.txt file
```bash
pip install -r requirements.txt
```

Run the main script

```bash
python3 main.py
```


#### HOW TO MAKE REQUEST

Go to http://127.0.0.1:8000

if the text "This is a testing page. It tells you this is working :)" showing then the server running.

Go to http://127.0.0.1/start?video_input=https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_20mb.mp4

If you see the result. Then it running

You edit the input to be RTSP connection with your CCTV
## Environment Variables

If you want to customize this, you can modified the following environment variables to your .env file


