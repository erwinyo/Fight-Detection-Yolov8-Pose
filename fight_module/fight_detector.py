import torch
import torch.nn as nn

from fight_module.util import *


class ThreeLayerClassifier(nn.Module):
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
    def __init__(self, fight_model, fps):
        # Architect the deep learning structure
        self.input_size = 16
        self.hidden_size = 8
        self.output_size = 1
        self.model = ThreeLayerClassifier(self.input_size, self.hidden_size, self.output_size)
        self.model.load_state_dict(torch.load(fight_model))
        self.model.eval()  # Set to evaluation mode

        # Coordinate for angel
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

        # Set up the thresholds
        self.threshold = 0.8  # Dictate how deep learning is sure there is fight on that frame
        self.conclusion_threshold = 2  # Dictate how hard the program conclude if a person is in fight action (1 - 3)
        self.FPS = fps

        # Event variables
        self.fight_detected = 0

    def detect(self, conf, xyn):
        input_list = []
        keypoint_unseen = False
        for n in self.coordinate_for_angel:
            # Keypoint number that we want to make new angel
            first, mid, end = n[0], n[1], n[2]

            # Gather the coordinate with keypoint number
            c1, c2, c3 = xyn[first], xyn[mid], xyn[end]
            # Check if all three coordinate of one key points is all zeros
            if is_coordinate_zero(c1, c2, c2):
                keypoint_unseen = True
                break
            else:
                # Getting angel from three coordinate
                input_list.append(calculate_angle(c1, c2, c3))
                # Getting the confs mean of three of those coordinate
                conf1, conf2, conf3 = conf[first], conf[mid], conf[end]
                input_list.append(torch.mean(torch.Tensor([conf1, conf2, conf3])).item())

        if keypoint_unseen:
            return

        # Make a prediction
        prediction = self.model(torch.Tensor(input_list))
        if prediction.item() > self.threshold:
            # FIGHT
            # this will grow exponentially according to number of person fighting on scene
            # if there is two person, and this will be added 2 for each frame
            self.fight_detected += 1
        else:
            # NO FIGHT
            # this if statement is for fight_detected not exceed negative value
            if self.fight_detected > 0:
                self.fight_detected -= self.conclusion_threshold
                # this value will decide how hard the program will conclude there is a fight in the frame
                # the higher the value, the more hard program to conclude

        # Threshold for fight_detected value, when it concludes there is fight on the frame
        # THRESHOLD = FPS * NUMBER OF PERSON DETECTED
        if self.fight_detected > self.FPS:
            return True
        else:
            return False
