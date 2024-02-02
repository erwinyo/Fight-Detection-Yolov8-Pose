import os

import torch
import torch.nn as nn
from fight_module.util import *
import dotenv

dotenv.load_dotenv()


class ThreeLayerClassifier(nn.Module):
    """
    Neural network model with three layers for classification.
    """

    def __init__(self, input_size, hidden_size1, output_size):
        super(ThreeLayerClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size1, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


class FightDetector:
    """
    Fight detection module using a deep learning model.
    """

    def __init__(self, fight_model):
        # Load pre-trained model
        self.input_size = 16
        self.hidden_size = 8
        self.output_size = 1
        self.model = ThreeLayerClassifier(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(torch.load(fight_model))
        self.model.eval()  # Set to evaluation mode

        # Define keypoints for calculating angles
        self.coordinate_for_angel = [
            [8, 6, 2],
            [11, 5, 7],
            [6, 8, 10],
            [5, 7, 9],
            [6, 12, 14],
            [5, 11, 13],
            [12, 14, 16],
            [11, 13, 15]
        ]

        # Set up detection thresholds
        self.threshold = float(os.getenv("THRESHOLD"))
        self.conclusion_threshold = float(os.getenv("CONCLUSION_THRESHOLD"))
        self.final_threshold = float(os.getenv("FINAL_THRESHOLD"))

        # Event variables
        self.fight_detected = 0

    def detect(self, conf, xyn):
        """
        Detects fight action based on keypoints and confidence scores.

        Args:
            conf (list): Confidence scores for each keypoint.
            xyn (list): Coordinates (x, y, visibility) for each keypoint.

        Returns:
            bool: True if fight detected, False otherwise.
        """
        input_list = []
        keypoint_unseen = False

        for n in self.coordinate_for_angel:
            # Keypoint numbers for creating angles
            first, mid, end = n[0], n[1], n[2]

            # Gather coordinates with keypoint numbers
            c1, c2, c3 = xyn[first], xyn[mid], xyn[end]

            # Check if all three coordinates of one keypoint are all zeros
            if is_coordinate_zero(c1, c2, c2):
                keypoint_unseen = True
                break
            else:
                # Calculate angle from three coordinates
                input_list.append(calculate_angle(c1, c2, c3))
                # Calculate the mean confidence score of the three coordinates
                conf1, conf2, conf3 = conf[first], conf[mid], conf[end]
                input_list.append(torch.mean(torch.Tensor([conf1, conf2, conf3])).item())

        if keypoint_unseen:
            return False

        # Make a prediction using the model
        prediction = self.model(torch.Tensor(input_list))

        # Update fight detection count
        if prediction.item() > self.threshold:
            self.fight_detected += 1
        else:
            # Decrease count when no fight is detected
            if self.fight_detected > 0:
                self.fight_detected -= self.conclusion_threshold

        # Check if a fight is concluded based on the detection count
        if self.fight_detected > self.final_threshold:
            return True
        else:
            return False
