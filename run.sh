#!/bin/bash

export model=model/yolo/yolov8n-pose.pt format=engine device=0

python3 main.py
