from ultralytics import YOLO


class YoloPoseEstimation:
    def __init__(self, yolo_model):
        self.model = YOLO(yolo_model)
        self.result = None

    def estimate(self, input):
        self.result = self.model.track(input, stream=True, tracker="bytetrack.yaml", conf=0.4, persist=True)
        return self.result

    def info(self):
        # Process results list
        for res in self.result:
            boxes = res.boxes  # Boxes object for bbox outputs
            masks = res.masks  # Masks object for segmentation masks outputs
            keypoints = res.keypoints  # Keypoints object for pose outputs
            probs = res.probs  # Probs object for classification outputs

            print(f"Boxes : {boxes}")
            print(f"Masks : {masks}")
            print(f"Keypoints : {keypoints}")
            print(f"Probs : {probs}")
